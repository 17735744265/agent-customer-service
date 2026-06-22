"""主入口：FastAPI 服务 + 命令行交互

v2.0 更新：
- 会话管理：通过 session_id 关联多轮对话历史
- 真流式输出：使用 graph.astream 节点级流式推送
- 错误防御：四层防御（参数校验→工具重试→降级回复→全局兜底）
"""

import os
import uuid
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ========== 初始化 LLM ==========
def get_llm():
    from config import LLM_BASE_URL, LLM_MODEL
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        logger.warning("未设置 DEEPSEEK_API_KEY，请在 .env 文件中配置")
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        api_key=api_key,
        temperature=0.7,
        streaming=True
    )


import config
config.llm = get_llm()


def init_vectorstore():
    """首次运行时构建向量数据库"""
    from rag.vectorstore import build_vectorstore
    if not os.path.exists(config.CHROMA_PERSIST_DIR):
        logger.info("首次运行，正在构建商品知识库...")
        build_vectorstore()
        logger.info("知识库构建完成")
    else:
        logger.info("商品知识库已就绪")


# ========== 会话管理器 ==========
class SessionManager:
    """内存会话管理器：通过 session_id 关联多轮对话历史
    
    核心能力：
    - 自动创建/复用会话
    - 滑动窗口截断历史消息（防止token超限）
    - 自动清理过期会话
    """

    def __init__(self, max_sessions: int = 100, max_history_rounds: int = 10):
        self.sessions: dict[str, dict] = {}
        self.max_sessions = max_sessions
        self.max_history_rounds = max_history_rounds

    def get_or_create(self, session_id: str = "") -> tuple[str, dict]:
        """获取已有会话或创建新会话，返回 (session_id, state)"""
        if not session_id:
            session_id = str(uuid.uuid4())[:8]

        if session_id not in self.sessions:
            # 会话数超限时清理最旧的
            if len(self.sessions) >= self.max_sessions:
                oldest = min(self.sessions, key=lambda k: self.sessions[k].get("last_active", ""))
                del self.sessions[oldest]
                logger.info(f"清理过期会话: {oldest}")

            self.sessions[session_id] = {
                "messages": [],
                "user_intent": "unknown",
                "requirements": {},
                "search_results": [],
                "comparison": "",
                "recommendation": "",
                "last_active": datetime.now().isoformat()
            }
            logger.info(f"新会话创建: {session_id}")

        self.sessions[session_id]["last_active"] = datetime.now().isoformat()
        return session_id, self.sessions[session_id]

    def add_user_message(self, session_id: str, content: str):
        """追加用户消息并做滑动窗口截断"""
        if session_id not in self.sessions:
            return
        state = self.sessions[session_id]
        state["messages"].append(HumanMessage(content=content))

        # 滑动窗口：超过max_history_rounds轮时只保留最近的
        max_messages = self.max_history_rounds * 2  # 每轮1条用户+1条助手
        if len(state["messages"]) > max_messages:
            state["messages"] = state["messages"][-max_messages:]

    def update_state(self, session_id: str, result: dict):
        """用图执行结果更新会话状态"""
        if session_id not in self.sessions:
            return
        for key in ["messages", "user_intent", "requirements", "search_results", "comparison", "recommendation"]:
            if key in result:
                self.sessions[session_id][key] = result[key]


session_manager = SessionManager()


# ========== 命令行交互模式 ==========
def chat_cli():
    """命令行交互：模拟多轮导购对话"""
    from agent.graph import build_graph

    init_vectorstore()
    graph = build_graph()

    print("=" * 50)
    print("智能售前导购Agent 已启动")
    print("输入你的购物需求，输入 quit 退出")
    print("=" * 50)

    session_id, state = session_manager.get_or_create("cli")

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("再见！")
            break
        if not user_input:
            continue

        session_manager.add_user_message(session_id, user_input)
        state = session_manager.sessions[session_id]

        result = graph.invoke(state)

        session_manager.update_state(session_id, result)

        if result.get("messages"):
            last_msg = result["messages"][-1]
            print(f"\n导购助手: {last_msg.content}")

        print(f"[意图: {result.get('user_intent', '?')}] [需求: {result.get('requirements', {})}]")


# ========== FastAPI 服务模式 ==========
def create_app():
    """创建 FastAPI 应用"""
    from fastapi import FastAPI, Query
    from fastapi.responses import StreamingResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    from agent.graph import build_graph

    init_vectorstore()
    graph = build_graph()

    app = FastAPI(title="智能售前导购Agent", version="2.0.0")

    # CORS（开发环境允许所有来源）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def index():
        """返回前端页面"""
        return FileResponse("static/index.html")

    @app.post("/chat")
    async def chat(
        message: str = Query(..., description="用户消息"),
        session_id: str = Query("", description="会话ID，为空则新建")
    ):
        """普通对话接口（支持多轮）"""
        sid, state = session_manager.get_or_create(session_id)
        session_manager.add_user_message(sid, message)
        state = session_manager.sessions[sid]

        try:
            result = graph.invoke(state)
        except Exception as e:
            logger.error(f"Agent执行失败: {e}")
            return {
                "reply": "抱歉，系统暂时无法处理您的请求，请稍后再试。",
                "session_id": sid,
                "intent": "error"
            }

        session_manager.update_state(sid, result)

        last_msg = result["messages"][-1] if result.get("messages") else None

        return {
            "reply": last_msg.content if last_msg else "抱歉，我无法处理这个请求。",
            "session_id": sid,
            "intent": result.get("user_intent", "unknown"),
            "requirements": result.get("requirements", {})
        }

    @app.post("/chat/stream")
    async def chat_stream(
        message: str = Query(..., description="用户消息"),
        session_id: str = Query("", description="会话ID，为空则新建")
    ):
        """流式对话接口（真流式：节点级实时推送）
        
        与旧版区别：
        - 旧版：graph.invoke等全部算完再逐字吐 → 用户体感等于没流
        - 新版：graph.astream每个节点完成即推送 → 用户实时看到进度
        - 推荐内容逐字符输出，体感接近token级流式
        """
        sid, state = session_manager.get_or_create(session_id)
        session_manager.add_user_message(sid, message)
        state = session_manager.sessions[sid]

        async def generate():
            try:
                from config import llm as _llm
                # 第一帧推送 session_id
                yield f"data: {json.dumps({'type': 'session', 'session_id': sid}, ensure_ascii=False)}\n\n"

                final_state = dict(state)
                token_stream_prompt = ""

                async for output in graph.astream(state):
                    for node_name, node_result in output.items():
                        yield f"data: {json.dumps({'type': 'progress', 'node': node_name}, ensure_ascii=False)}\n\n"

                        if node_name == "clarify":
                            msgs = node_result.get("messages", [])
                            if msgs:
                                content = msgs[-1].content if hasattr(msgs[-1], 'content') else str(msgs[-1])
                                yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"

                        elif node_name == "followup":
                            # 跟进对话也直接推送，不走token流式
                            msgs = node_result.get("messages", [])
                            if msgs:
                                content = msgs[-1].content if hasattr(msgs[-1], 'content') else str(msgs[-1])
                                yield f"data: {json.dumps({'content': content}, ensure_ascii=False)}\n\n"

                        elif node_name == "recommend":
                            # 推荐节点不直接推送，改为后续token级流式
                            # 但先把数据收集到final_state
                            pass

                        elif node_name == "understand":
                            intent = node_result.get("user_intent", "unknown")
                            yield f"data: {json.dumps({'type': 'intent', 'intent': intent}, ensure_ascii=False)}\n\n"

                        # 统一收集节点结果到final_state
                        for key in ["messages", "user_intent", "requirements", "search_results", "comparison", "recommendation"]:
                            if key in node_result:
                                if key == "messages":
                                    msgs = node_result[key]
                                    if isinstance(msgs, list):
                                        final_state.setdefault("messages", []).extend(msgs)
                                    else:
                                        final_state.setdefault("messages", []).append(msgs)
                                else:
                                    final_state[key] = node_result[key]

                # 推荐内容逐token流式推送
                rec_text = final_state.get("recommendation", "")
                intent = final_state.get("user_intent", "vague")
                req = final_state.get("requirements", {})

                # 格式化对话历史
                msgs = final_state.get("messages", [])
                start = max(0, len(msgs) - 10)
                history_lines = []
                for m in msgs[start:]:
                    role = "用户" if m.__class__.__name__ == "HumanMessage" else "助手"
                    history_lines.append(f"{role}: {m.content}")
                history = "\n".join(history_lines)

                # 先构造推荐prompt
                if intent == "compare":
                    from agent.prompts import COMPARE_PROMPT
                    token_stream_prompt = COMPARE_PROMPT.format(
                        comparison_data=final_state.get("comparison", ""),
                        requirements=json.dumps(req, ensure_ascii=False),
                        history=history
                    )
                else:
                    from agent.prompts import RECOMMEND_PROMPT
                    search_results = final_state.get("search_results", [])
                    results_text = ""
                    for i, p in enumerate(search_results[:3], 1):
                        if isinstance(p, dict) and "name" in p:
                            results_text += f"{i}. {p['name']} - ¥{p.get('price', '未知')}\n   亮点: {p.get('highlights', '')}\n   适合: {p.get('target_user', '')}\n\n"
                        else:
                            results_text += f"{i}. {p}\n"
                    token_stream_prompt = RECOMMEND_PROMPT.format(
                        requirements=json.dumps(req, ensure_ascii=False),
                        search_results=results_text or "暂无匹配商品",
                        history=history
                    )

                # 尝试LLM astream逐token输出
                stream_ok = False
                try:
                    full_rec = ""
                    async for chunk in _llm.astream(token_stream_prompt):
                        token = chunk.content
                        if token:
                            full_rec += token
                            yield f"data: {json.dumps({'content': token}, ensure_ascii=False)}\n\n"
                            stream_ok = True
                    if stream_ok:
                        final_state["recommendation"] = full_rec
                        if final_state.get("messages"):
                            for i in range(len(final_state["messages"]) - 1, -1, -1):
                                if isinstance(final_state["messages"][i], AIMessage):
                                    final_state["messages"][i] = AIMessage(content=full_rec)
                                    break
                except Exception as e:
                    logger.warning(f"LLM astream失败: {e}")

                # LLM流式失败时，回退到graph兜底结果
                if not stream_ok and rec_text:
                    for i in range(0, len(rec_text), 2):
                        yield f"data: {json.dumps({'content': rec_text[i:i+2]}, ensure_ascii=False)}\n\n"

                session_manager.update_state(sid, final_state)

            except Exception as e:
                logger.error(f"流式输出失败: {e}")
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/health")
    async def health():
        return {"status": "ok", "active_sessions": len(session_manager.sessions)}

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="智能售前导购Agent")
    parser.add_argument("--mode", choices=["cli", "api"], default="cli", help="运行模式")
    parser.add_argument("--port", type=int, default=8000, help="API服务端口")
    args = parser.parse_args()

    if args.mode == "cli":
        chat_cli()
    else:
        import uvicorn
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=args.port)
