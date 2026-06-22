# ========== LLM 配置 ==========
# 使用 DeepSeek API（兼容 OpenAI 格式）
LLM_BASE_URL = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"
LLM_API_KEY = ""  # 填你的 DeepSeek API Key，或设置环境变量 DEEPSEEK_API_KEY

# ========== 向量数据库配置 ==========
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"  # 中文embedding模型，本地运行

# ========== RAG 配置 ==========
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_TOP_K = 3

# ========== 商品数据 ==========
PRODUCTS_FILE = "./data/products.json"
