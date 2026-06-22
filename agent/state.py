"""LangGraph 状态定义"""

from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage
import operator


class ShoppingState(TypedDict):
    """售前导购Agent的状态
    
    核心字段说明：
    - messages: 对话历史，用 operator.add 做追加
    - user_intent: 用户意图分类（specific/vague/compare/unknown）
    - requirements: 提取的用户需求
    - search_results: 商品检索结果
    - comparison: 对比分析结果
    - recommendation: 最终推荐
    """
    messages: Annotated[list[BaseMessage], operator.add]
    user_intent: str  # specific / vague / compare / followup / unknown
    requirements: dict  # 提取的需求：{"category": "手机", "budget": "3000-4000", ...}
    search_results: list[dict]  # 检索到的商品列表
    comparison: str  # 对比分析文本
    recommendation: str  # 推荐结果文本
