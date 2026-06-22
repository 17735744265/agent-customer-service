"""LangGraph 状态图编排：售前导购工作流

工作流：
用户输入 → 理解需求 → 条件路由
  ├─ specific（需求明确）→ 商品检索 → 生成推荐
  ├─ vague（需求模糊）  → 追问澄清 → 回到理解需求
  └─ compare（对比需求）→ 商品检索 → 对比分析 → 生成推荐
"""

from langgraph.graph import StateGraph, END
from agent.state import ShoppingState
from agent.nodes import (
    understand_need,
    ask_clarification,
    handle_followup,
    search_products_node,
    compare_products_node,
    generate_recommendation,
    route_by_intent,
)


def build_graph() -> StateGraph:
    """构建售前导购Agent的状态图"""
    
    # 1. 创建状态图
    graph = StateGraph(ShoppingState)
    
    # 2. 添加节点（每个节点对应一个处理步骤）
    graph.add_node("understand", understand_need)          # 理解需求+意图分类
    graph.add_node("clarify", ask_clarification)           # 追问澄清
    graph.add_node("followup", handle_followup)            # 跟进对话
    graph.add_node("search", search_products_node)         # 商品检索
    graph.add_node("compare", compare_products_node)       # 对比分析
    graph.add_node("recommend", generate_recommendation)   # 生成推荐
    
    # 3. 设置入口节点
    graph.set_entry_point("understand")
    
    # 4. 添加条件边：根据意图路由
    graph.add_conditional_edges(
        "understand",
        route_by_intent,
        {
            "specific": "search",     # 需求明确 → 直接检索
            "vague": "clarify",       # 需求模糊 → 追问
            "compare": "search",      # 对比需求 → 先检索再对比
            "followup": "followup",   # 跟进对话 → 直接回答
        }
    )
    
    # 5. 添加固定边
    graph.add_edge("clarify", END)            # 追问后等待用户回复
    graph.add_edge("followup", END)          # 跟进回答后结束本轮
    graph.add_edge("search", "recommend")     # 检索完 → 推荐（specific路径）
    graph.add_edge("compare", "recommend")    # 对比完 → 推荐（compare路径）
    graph.add_edge("recommend", END)          # 推荐完 → 结束本轮
    
    # 6. 编译图
    return graph.compile()
