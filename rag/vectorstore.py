"""RAG 商品知识库：向量化商品数据 + 语义检索"""

import json
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, PRODUCTS_FILE, RETRIEVAL_TOP_K


def get_embeddings():
    """获取本地 Embedding 模型"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def load_product_documents() -> list[Document]:
    """将商品数据转为 LangChain Document 格式"""
    with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
        products = json.load(f)
    
    documents = []
    for p in products:
        # 把商品信息拼成一段文本用于向量化
        specs_text = "\n".join([f"- {k}: {v}" for k, v in p["specs"].items()])
        content = f"""商品名称: {p['name']}
品类: {p['category']}
品牌: {p['brand']}
价格: ¥{p['price']}
核心亮点: {p['highlights']}
适合人群: {p['target_user']}
规格参数:
{specs_text}
常见问题: {p['faq']}"""
        
        doc = Document(
            page_content=content,
            metadata={
                "id": p["id"],
                "name": p["name"],
                "category": p["category"],
                "brand": p["brand"],
                "price": p["price"]
            }
        )
        documents.append(doc)
    
    return documents


def build_vectorstore():
    """构建向量数据库（首次运行或需要重建时调用）"""
    documents = load_product_documents()
    embeddings = get_embeddings()
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="products",
        persist_directory=CHROMA_PERSIST_DIR
    )
    
    print(f"向量数据库构建完成，共 {len(documents)} 条商品数据")
    return vectorstore


def get_vectorstore():
    """获取已有的向量数据库"""
    embeddings = get_embeddings()
    return Chroma(
        collection_name="products",
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )


def search_knowledge(query: str, top_k: int = RETRIEVAL_TOP_K) -> list[Document]:
    """语义检索商品知识库
    
    使用 MMR 策略：兼顾相关性与多样性
    
    Args:
        query: 查询文本
        top_k: 返回结果数量
    
    Returns:
        匹配的 Document 列表
    """
    vectorstore = get_vectorstore()
    
    results = vectorstore.max_marginal_relevance_search(
        query=query,
        k=top_k,
        fetch_k=top_k * 3  # 先多取，再 MMR 筛选
    )
    
    return results


if __name__ == "__main__":
    # 单独运行此文件可构建向量数据库
    build_vectorstore()
    # 测试检索
    results = search_knowledge("拍照好的手机")
    for r in results:
        print(f"- {r.metadata['name']}: {r.page_content[:80]}...")
