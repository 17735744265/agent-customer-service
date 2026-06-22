"""Agent 使用的工具函数"""

import json
from langchain_core.tools import tool


def _load_products():
    """加载商品数据"""
    with open("./data/products.json", "r", encoding="utf-8") as f:
        return json.load(f)


@tool
def search_products(query: str, category: str = "", budget_max: int = 0) -> list[dict]:
    """根据关键词、品类和预算搜索商品
    
    Args:
        query: 搜索关键词，如"拍照手机""轻薄笔记本"
        category: 商品品类，如"手机""笔记本""平板"，为空则不限
        budget_max: 最大预算，0表示不限
    
    Returns:
        匹配的商品列表
    """
    products = _load_products()
    results = []
    
    for p in products:
        # 品类过滤
        if category and p["category"] != category:
            continue
        # 预算过滤
        if budget_max > 0 and p["price"] > budget_max:
            continue
        # 关键词匹配（搜索名称、亮点、目标用户）
        searchable = f"{p['name']} {p['highlights']} {p['target_user']} {p['brand']}"
        if query.lower() in searchable.lower() or not query:
            results.append(p)
    
    return results[:5]  # 最多返回5个


@tool
def get_product_details(product_name: str) -> dict:
    """获取商品详细信息，包括规格参数和常见问题
    
    Args:
        product_name: 商品名称，如"小米14"
    
    Returns:
        商品详细信息
    """
    products = _load_products()
    for p in products:
        if product_name in p["name"]:
            return p
    return {"error": f"未找到商品: {product_name}"}


@tool
def compare_products(name_a: str, name_b: str) -> str:
    """对比两个商品的规格参数
    
    Args:
        name_a: 商品A名称
        name_b: 商品B名称
    
    Returns:
        对比结果文本
    """
    products = _load_products()
    pa, pb = None, None
    for p in products:
        if name_a in p["name"]:
            pa = p
        if name_b in p["name"]:
            pb = p
    
    if not pa:
        return f"未找到商品: {name_a}"
    if not pb:
        return f"未找到商品: {name_b}"
    
    lines = [f"## {pa['name']} vs {pb['name']}\n"]
    lines.append(f"| 项目 | {pa['name']} | {pb['name']} |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| 价格 | ¥{pa['price']} | ¥{pb['price']} |")
    
    # 合并所有规格项
    all_specs = set(pa["specs"].keys()) | set(pb["specs"].keys())
    for key in all_specs:
        va = pa["specs"].get(key, "-")
        vb = pb["specs"].get(key, "-")
        lines.append(f"| {key} | {va} | {vb} |")
    
    return "\n".join(lines)


# 所有工具列表，注册到 Agent 时使用
ALL_TOOLS = [search_products, get_product_details, compare_products]
