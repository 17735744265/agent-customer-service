"""LangGraph 图节点：每个节点对应工作流中的一个步骤

四层错误防御：
1. 参数校验：检查工具调用参数完整性
2. 工具重试：失败后自动重试（最多2次）
3. 降级回复：重试失败后用已有信息生成不依赖该工具的回答
4. 全局兜底：所有环节都失败时返回友好提示
"""

import json
import logging
from langchain_core.messages import AIMessage
from agent.state import ShoppingState
from agent.tools import search_products, get_product_details, compare_products
from agent.prompts import UNDERSTAND_PROMPT, CLARIFY_PROMPT, RECOMMEND_PROMPT, COMPARE_PROMPT
from config import llm

logger = logging.getLogger(__name__)


# ========== 四层错误防御工具函数 ==========

def validate_params(params: dict, required_keys: list[str]) -> tuple[bool, str]:
    """第1层：参数校验，检查必填参数是否存在且非空"""
    for key in required_keys:
        if key not in params or params[key] is None:
            return False, f"缺少必要参数: {key}"
        if isinstance(params[key], str) and not params[key].strip():
            return False, f"参数 {key} 不能为空字符串"
    return True, ""


def safe_tool_call(tool, params: dict, required_keys: list[str], max_retries: int = 2):
    """安全工具调用，封装第1-3层防御

    第1层：参数校验 → 不通过直接返回空
    第2层：工具重试 → 失败后自动重试max_retries次
    第3层：降级回复 → 全部重试失败返回空结果，由调用方处理
    """
    # 第1层：参数校验
    valid, err_msg = validate_params(params, required_keys)
    if not valid:
        logger.warning(f"参数校验失败: {err_msg}, params={params}")
        return None

    # 第2层：工具重试
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = tool.invoke(params)
            return result
        except Exception as e:
            last_error = e
            logger.warning(f"工具 {tool.name} 第{attempt+1}次调用失败: {e}")
            if attempt < max_retries:
                import time
                time.sleep(1)

    # 第3层：降级回复 — 返回None，由调用方决定如何降级
    logger.error(f"工具 {tool.name} 重试{max_retries}次后仍失败: {last_error}")
    return None


# ========== 图节点 ==========

def _format_history(state: ShoppingState, max_rounds: int = 5) -> str:
    """把对话历史格式化为文本，最近max_rounds轮"""
    messages = state.get("messages", [])
    # 每轮1条用户+1条助手
    start = max(0, len(messages) - max_rounds * 2)
    lines = []
    for msg in messages[start:]:
        role = "用户" if msg.__class__.__name__ == "HumanMessage" else "助手"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def understand_need(state: ShoppingState) -> dict:
    """节点1：理解用户需求 + 意图分类"""
    history = _format_history(state)
    last_message = state["messages"][-1].content

    prompt = UNDERSTAND_PROMPT.format(user_input=last_message, history=history)

    # 第4层：LLM调用失败时兜底
    try:
        response = llm.invoke(prompt).content
    except Exception as e:
        logger.error(f"LLM调用失败: {e}")
        return {"user_intent": "vague", "requirements": {}}

    intent = "vague"
    requirements = {}

    # 更鲁棒的解析：不依赖严格格式
    try:
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("INTENT:"):
                intent_str = line.split(":", 1)[1].strip().lower()
                if intent_str in ["specific", "vague", "compare", "followup"]:
                    intent = intent_str
            elif line.upper().startswith("REQUIREMENTS:"):
                req_str = line.split(":", 1)[1].strip()
                try:
                    parsed = json.loads(req_str)
                    requirements = parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    requirements = {}
    except Exception as e:
        logger.warning(f"解析LLM输出失败: {e}, 原始输出: {response[:200]}")

    return {"user_intent": intent, "requirements": requirements}


def handle_followup(state: ShoppingState) -> dict:
    """跟进对话节点：处理用户对之前推荐的追问"""
    from agent.prompts import FOLLOWUP_PROMPT
    history = _format_history(state)
    last_message = state["messages"][-1].content
    recommendation = state.get("recommendation", "")

    prompt = FOLLOWUP_PROMPT.format(
        history=history,
        recommendation=recommendation,
        user_input=last_message
    )

    try:
        response = llm.invoke(prompt).content
    except Exception as e:
        logger.error(f"跟进对话LLM调用失败: {e}")
        response = "您可以去京东、天猫或品牌官方门店购买，经常有优惠活动。还有什么想了解的吗？"

    return {"messages": [AIMessage(content=response)]}


def ask_clarification(state: ShoppingState) -> dict:
    """节点2：追问澄清"""
    requirements = state.get("requirements", {})
    history = _format_history(state)

    prompt = CLARIFY_PROMPT.format(requirements=json.dumps(requirements, ensure_ascii=False), history=history)

    try:
        response = llm.invoke(prompt).content
    except Exception as e:
        logger.error(f"LLM调用失败: {e}")
        response = "能告诉我您想看什么品类的商品吗？预算大概多少？"

    return {"messages": [AIMessage(content=response)]}


def search_products_node(state: ShoppingState) -> dict:
    """节点3：商品检索"""
    req = state.get("requirements", {})
    category = req.get("category", "")
    budget_max = req.get("budget_max", 0)
    keywords = " ".join(req.get("keywords", []))
    # keywords为空时用category兜底，避免query为空字符串
    query_text = keywords or category or "手机"

    # 安全工具调用（含参数校验+重试+降级）
    results = safe_tool_call(
        search_products,
        {"query": query_text, "category": category, "budget_max": budget_max},
        required_keys=["query"]
    )

    # 降级处理：工具返回None或空
    if results is None:
        results = []
        logger.info("商品检索工具失败，尝试RAG降级")
    elif isinstance(results, dict) and "error" in results:
        results = []
    elif not isinstance(results, list):
        results = list(results) if results else []

    # RAG补充检索（工具检索不足时）
    if len(results) < 2:
        try:
            from rag.vectorstore import search_knowledge
            rag_results = search_knowledge(f"{category} {keywords}")
            for doc in rag_results:
                results.append({
                    "name": doc.metadata.get("name", ""),
                    "rag_info": doc.page_content
                })
        except Exception as e:
            logger.warning(f"RAG检索也失败，只能依赖已有结果: {e}")

    return {"search_results": results}


def compare_products_node(state: ShoppingState) -> dict:
    """节点4：商品对比"""
    req = state.get("requirements", {})
    last_message = state["messages"][-1].content

    compare_result = ""

    # 尝试从用户输入提取商品名
    import re
    pattern = r"(.+?)(?:和|与|对比|vs)(.+?)(?:的|相比|哪个|区别|对比|$)"
    match = re.search(pattern, last_message)

    if match:
        name_a = match.group(1).strip()
        name_b = match.group(2).strip()
        result = safe_tool_call(
            compare_products,
            {"name_a": name_a, "name_b": name_b},
            required_keys=["name_a", "name_b"]
        )
        if isinstance(result, str) and result:
            compare_result = result

    # 正则没匹配到，从检索结果取前两个
    if not compare_result:
        results = state.get("search_results", [])
        if len(results) >= 2:
            name_a = results[0].get("name", "")
            name_b = results[1].get("name", "")
            if name_a and name_b:
                result = safe_tool_call(
                    compare_products,
                    {"name_a": name_a, "name_b": name_b},
                    required_keys=["name_a", "name_b"]
                )
                if isinstance(result, str) and result:
                    compare_result = result

    # 第4层兜底：对比完全失败
    if not compare_result:
        compare_result = "无法识别要对比的商品，请明确指定两个商品名称，例如：小米14和iPhone 15 Pro哪个好？"

    return {"comparison": compare_result}


def generate_recommendation(state: ShoppingState) -> dict:
    """节点5：生成推荐"""
    intent = state.get("user_intent", "vague")
    req = state.get("requirements", {})
    history = _format_history(state)

    try:
        if intent == "compare":
            comparison = state.get("comparison", "")
            prompt = COMPARE_PROMPT.format(
                comparison_data=comparison,
                requirements=json.dumps(req, ensure_ascii=False),
                history=history
            )
        else:
            search_results = state.get("search_results", [])
            results_text = ""
            for i, p in enumerate(search_results[:3], 1):
                if isinstance(p, dict) and "name" in p:
                    results_text += f"{i}. {p['name']} - ¥{p.get('price', '未知')}\n   亮点: {p.get('highlights', '')}\n   适合: {p.get('target_user', '')}\n\n"
                else:
                    results_text += f"{i}. {p}\n"

            prompt = RECOMMEND_PROMPT.format(
                requirements=json.dumps(req, ensure_ascii=False),
                search_results=results_text or "暂无匹配商品",
                history=history
            )

        response = llm.invoke(prompt).content

    except Exception as e:
        # 第4层全局兜底：LLM调用失败时用工具返回的原始数据拼推荐
        logger.error(f"推荐生成失败: {e}")
        search_results = state.get("search_results", [])
        if search_results:
            fallback_items = []
            for p in search_results[:3]:
                if isinstance(p, dict) and "name" in p:
                    fallback_items.append(f"- {p['name']} ¥{p.get('price', '未知')} — {p.get('highlights', '')}")
            response = "为您找到以下商品：\n" + "\n".join(fallback_items) + "\n如需了解更多请告诉我。"
        else:
            response = "抱歉，暂时无法生成推荐，请稍后再试。"

    return {
        "recommendation": response,
        "messages": [AIMessage(content=response)]
    }


# ========== 条件路由函数 ==========
def route_by_intent(state: ShoppingState) -> str:
    """根据用户意图路由到不同节点"""
    return state.get("user_intent", "vague")
