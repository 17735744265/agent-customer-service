---
AIGC:
    Label: "1"
    ContentProducer: 001191110102MACQD9K64018705
    ProduceID: 2486413884662268_0/project_7652647328645677375-files/README_shopping_agent.md
    ReservedCode1: ""
    ContentPropagator: 001191110102MACQD9K64028705
    PropagateID: 2486413884662268#1782116883513
    ReservedCode2: ""
---
# 智能售前导购 Agent

基于 LangGraph 的电商智能导购助手，支持需求理解、追问澄清、商品推荐、对比分析和跟进对话，具备真流式输出和多轮会话管理能力。

## 项目架构

```
售前导购Agent/
├── main.py                 # 主入口：FastAPI 服务 + CLI 交互
├── config.py               # 全局配置（LLM / RAG / 商品数据路径）
├── requirements.txt        # 依赖列表
├── .env                    # 环境变量（DEEPSEEK_API_KEY）
├── agent/
│   ├── state.py            # ShoppingState 状态定义
│   ├── graph.py            # LangGraph 状态图编排
│   ├── nodes.py            # 图节点（5 个处理节点 + 路由函数）
│   ├── prompts.py          # Prompt 模板（5 套）
│   └── tools.py            # 工具函数（搜索 / 详情 / 对比）
├── rag/
│   └── vectorstore.py      # ChromaDB 向量存储 + MMR 语义检索
├── data/
│   └── products.json       # 商品数据（6 条）
├── static/
│   └── index.html          # 前端页面（SSE 流式接收）
└── chroma_db/              # 已构建的向量数据库（首次运行自动生成）
```

## 工作流

```
用户输入
  │
  ▼
understand（理解需求 + 意图分类）
  │
  ├─ specific ──→ search（商品检索）──→ recommend（生成推荐）
  ├─ vague ─────→ clarify（追问澄清）──→ END
  ├─ compare ───→ search ──→ compare（对比分析）──→ recommend
  └─ followup ──→ followup（跟进回答）──→ END
```

四种意图类型：

| 意图 | 触发场景 | 处理路径 |
|------|----------|----------|
| specific | "预算2000拍风景的手机" | 检索 → 推荐 |
| vague | "想买个手机" | 追问品类/预算 |
| compare | "小米14和iPhone哪个好" | 检索 → 对比 → 推荐 |
| followup | "还行""怎么购买" | 直接回答，不重新搜索 |

## 核心特性

### 四层错误防御

| 层级 | 策略 | 示例 |
|------|------|------|
| 第 1 层 | 参数校验 | query 为空时用 category 兜底 |
| 第 2 层 | 工具重试 | search_products 失败自动重试 2 次 |
| 第 3 层 | 降级回复 | 工具全部失败后 RAG 语义检索补充 |
| 第 4 层 | 全局兜底 | LLM 调用失败返回友好提示 |

### 多轮会话管理

- 通过 `session_id` 关联对话历史
- 滑动窗口截断（默认保留最近 10 轮，防止 token 超限）
- 过期会话自动清理（最大 100 个并发会话）
- 前端 localStorage 持久化 session_id，刷新页面不丢失上下文

### 真流式输出

- 后端：`graph.astream` 节点级流式 + `_llm.astream` token 级流式
- 前端：SSE（Server-Sent Events）逐 token 追加渲染
- 推荐内容逐字符推送，追问/跟进节点整条推送
- 流式失败时自动回退到普通接口

### RAG 语义检索

- ChromaDB 向量存储，中文 text2vec-base-chinese Embedding
- MMR（最大边际相关性）检索策略，兼顾相关性与多样性
- 关键词搜索结果不足时自动 RAG 降级补充

### 对话历史注入

- 所有 Prompt 模板注入最近 5 轮对话历史
- 跟进对话节点读取历史推荐结果，避免重复搜索

## 快速开始

### 环境要求

- Python 3.10+
- DeepSeek API Key（兼容 OpenAI 格式）

### 安装

```bash
git clone <repo-url>
cd 售前导购Agent
pip install -r requirements.txt
```

### 配置

在项目根目录创建 `.env` 文件：

```env
DEEPSEEK_API_KEY=sk-your-key-here
```

或在 `config.py` 中直接填写 `LLM_API_KEY`。

### 运行

**CLI 模式**（命令行交互）：

```bash
python main.py --mode cli
```

**API 模式**（Web 服务）：

```bash
python main.py --mode api
# 访问 http://localhost:8000
```

首次运行会自动构建 ChromaDB 向量数据库，后续启动直接加载。

## API 接口

### POST /chat

普通对话接口，返回完整 JSON 结果。

```
POST /chat?message=手机&session_id=abc123
```

响应：

```json
{
  "reply": "为您推荐以下商品...",
  "session_id": "abc123",
  "intent": "specific",
  "requirements": {"category": "手机", "budget_max": 0}
}
```

### POST /chat/stream

流式对话接口，SSE 逐 token 推送。

```
POST /chat/stream?message=手机&session_id=abc123
```

SSE 事件流：

```
data: {"type": "session", "session_id": "abc123"}
data: {"type": "progress", "node": "understand"}
data: {"type": "intent", "intent": "specific"}
data: {"type": "progress", "node": "search"}
data: {"type": "progress", "node": "recommend"}
data: {"content": "为"}
data: {"content": "您"}
data: {"content": "推荐"}
...
data: [DONE]
```

### GET /health

健康检查接口。

```json
{"status": "ok", "active_sessions": 3}
```

## 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| Agent 框架 | LangGraph | >=0.2.0 |
| LLM | DeepSeek Chat（OpenAI 兼容） | - |
| Embedding | text2vec-base-chinese（本地） | - |
| 向量数据库 | ChromaDB | >=0.4.0 |
| Web 框架 | FastAPI + Uvicorn | >=0.110.0 |
| 前端 | 原生 JS + SSE | - |

## 商品数据

当前包含 6 条商品数据（`data/products.json`）：

| 商品 | 品类 | 价格 | 亮点 |
|------|------|------|------|
| 小米14 | 手机 | ¥3,999 | 徕卡影像，小屏旗舰 |
| iPhone 15 Pro | 手机 | ¥8,999 | A17 Pro，钛金属 |
| Redmi Note 13 Pro | 手机 | ¥1,499 | 2 亿像素，千元性价比 |
| MacBook Air M3 | 笔记本 | ¥8,999 | M3 芯片，18h 续航 |
| 联想小新Pro16 | 笔记本 | ¥5,499 | 大屏高刷，Ultra 处理器 |
| 华为MatePad Pro 13.2 | 平板 | ¥5,199 | OLED 大屏，星闪手写 |

## 项目状态

### 已完成

- 四层错误防御（参数校验 / 工具重试 / RAG 降级 / 全局兜底）
- 多轮会话管理（session_id + 滑动窗口 + 过期清理）
- 真流式输出（节点级 + token 级双层流式）
- followup 意图识别与跟进对话节点
- 对话历史注入（5 轮上下文传入所有 Prompt）
- 前端 SSE 流式接收 + session_id 持久化
- 空查询兜底 + 字段安全取值

### 待优化

- Pydantic 校验 State（当前为 TypedDict，无字段约束）
- JWT 认证 + 滑动窗口限流
- JSON 格式日志 + trace_id 追踪
- 前后端接口类型对齐
- 商品数据增量更新机制
- 单元测试 / 集成测试

---

> 本内容由 Coze AI 生成，请遵循相关法律法规及《人工智能生成合成内容标识办法》使用与传播。
