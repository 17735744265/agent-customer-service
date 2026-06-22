"""主入口：FastAPI 服务 + 命令行交互"""

import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# ========== 初始化 LLM ==========
def get_llm():
    from config import LLM_BASE_URL, LLM_MODEL
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("警告: 未设置 DEEPSEEK_API_KEY，请在 .env 文件中配置")
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        api_key=api_key,
        temperature=0.7,
        streaming=True
    )


# 把 llm 挂到 config 上，供 nodes.py 导入
import config
config.llm = get_llm()


def init_vectorstore():
    """首次运行时构建向量数据库"""
    from rag.vectorstore import build_vectorstore
    import os
    if not os.path.exists(config.CHROMA_PERSIST_DIR):
        print("首次运行，正在构建商品知识库...")
        build_vectorstore()
        print("知识库构建完成！")
    else:
        print("商品知识库已就绪")


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
    
    # 多轮对话状态
    state = {
        "messages": [],
        "user_intent": "unknown",
        "requirements": {},
        "search_results": [],
        "comparison": "",
        "recommendation": ""
    }
    
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("再见！")
            break
        if not user_input:
            continue
        
        # 追加用户消息到状态
        state["messages"].append(HumanMessage(content=user_input))
        
        # 执行状态图
        result = graph.invoke(state)
        
        # 更新状态（保留对话历史）
        state["messages"] = result.get("messages", state["messages"])
        
        # 输出Agent回复
        if result.get("messages"):
            last_msg = result["messages"][-1]
            print(f"\n导购助手: {last_msg.content}")
        
        # 显示内部状态（调试用，可注释）
        print(f"[意图: {result.get('user_intent', '?')}] [需求: {result.get('requirements', {})}]")


# ========== FastAPI 服务模式 ==========
def create_app():
    """创建 FastAPI 应用"""
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from agent.graph import build_graph
    import json
    
    init_vectorstore()
    graph = build_graph()
    
    app = FastAPI(title="智能售前导购Agent", version="1.0.0")
    
    @app.post("/chat")
    async def chat(message: str):
        """普通对话接口
        
        Args:
            message: 用户消息
        
        Returns:
            Agent 回复
        """
        state = {
            "messages": [HumanMessage(content=message)],
            "user_intent": "unknown",
            "requirements": {},
            "search_results": [],
            "comparison": "",
            "recommendation": ""
        }
        
        result = graph.invoke(state)
        last_msg = result["messages"][-1] if result.get("messages") else None
        
        return {
            "reply": last_msg.content if last_msg else "抱歉，我无法处理这个请求。",
            "intent": result.get("user_intent", "unknown"),
            "requirements": result.get("requirements", {})
        }
    
    @app.post("/chat/stream")
    async def chat_stream(message: str):
        """流式输出接口
        
        Args:
            message: 用户消息
        
        Returns:
            SSE 流式响应
        """
        state = {
            "messages": [HumanMessage(content=message)],
            "user_intent": "unknown",
            "requirements": {},
            "search_results": [],
            "comparison": "",
            "recommendation": ""
        }
        
        def generate():
            # 先执行状态图拿到完整结果，再流式输出
            result = graph.invoke(state)
            last_msg = result["messages"][-1] if result.get("messages") else None
            if last_msg:
                # 逐字输出模拟流式效果
                content = last_msg.content
                for i in range(0, len(content), 5):
                    yield f"data: {json.dumps({'content': content[i:i+5]}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    
    return app


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="智能售前导购Agent")
    parser.add_argument("--mode", choices=["cli", "api"], default="cli", help="运行模式: cli=命令行, api=FastAPI服务")
    parser.add_argument("--port", type=int, default=8000, help="API服务端口")
    args = parser.parse_args()
    
    if args.mode == "cli":
        chat_cli()
    else:
        import uvicorn
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=args.port)
