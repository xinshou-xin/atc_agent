# Drug ATC Mapping Agent

药物 ATC 编码智能映射系统，基于 LLM + RAG 架构，自动将药物名称映射到 WHO ATC 分类编码。

## 项目结构

```
drug_Agent - 副本/
├── test-deepagents.py      # Deep Agents 运行入口
├── build_vector_db.py      # 重建本地 RAG 向量库（Chroma 分片）
├── config/
│   └── settings.py         # API Key、URL、参数、向量库配置
├── agents/
│   └── deep_atc_agent.py   # Deep Agent（ATC 推理，含反思与精化）
├── tools/
│   ├── deep_tools.py       # 6 个工具（含 search_local_atc_knowledge）
│   ├── drug_normalizer.py  # 药物名标准化（LLM 翻译 + INN）
│   ├── atc_search.py       # WHO ATC 网站搜索
│   ├── pubchem.py          # PubChem API
│   └── tavily.py           # Tavily 搜索 API
├── retriever/
│   ├── vector_db.py        # 向量库封装（Chroma 分片 / FAISS / NumPy）
│   ├── embedding.py        # bge-m3 Embedding
│   └── reranker.py         # bge-reranker 重排序
├── skills/                 # Deep Agents 领域规则
├── tests/                  # 单元测试
├── utils/                  # 日志、缓存
├── model/                  # 本地 bge-m3 / bge-reranker 模型
├── data/
│   ├── knowledge/          # ATC 知识源（6807 条）
│   └── vector_db/          # Chroma 分片向量库（17 分片）
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

编辑 `.env`：填 LLM API Key，向量库配置保持：

```bash
VECTOR_DB_PROVIDER=chroma
EMBEDDING_MODEL_PATH=.../model/BAAI/bge-m3
RERANKER_MODEL_PATH=.../model/BAAI/bge-reranker-v2-m3
```

Embedding / Reranker 支持三种模式：`local`（本地加载模型）、`api`（调用服务器上自建的
BGE-M3 / BGE-Reranker FastAPI 服务，本地不占内存，见 `.env` 中的 `EMBEDDING_API_URL` /
`RERANKER_API_URL`）、`openai` / `cohere`（商业 API）。本地显存/内存不足时把
`EMBEDDING_PROVIDER`、`RERANKER_PROVIDER` 设为 `api` 即可。

### 3. 重建本地 RAG 向量库（首次或数据变更时）

```bash
python build_vector_db.py
```

说明：本机 chromadb 对超大单集合的 HNSW 索引无法落盘，故按 400 条/片
分片存储（检索时合并），灌库后需等待 compactor 完成落盘（脚本内等待 300 秒）。

### 4. 运行

```bash
python test-deepagents.py
```

## 核心流程

1. **药物名称标准化** — LLM 翻译 + INN 通用名解析（translate_drug_name）
2. **多源信息检索** — PubChem / Tavily / 向量库
3. **ATC 编码推理** — 基于检索信息由 LLM 推理 ATC 编码
4. **反思验证** — Reflection Agent 验证并修正结果

## 支持的 LLM

- DeepSeek
- OpenAI / GPT
- Qwen（通义千问）
- 任何 OpenAI 兼容 API

## 支持的向量数据库

- ChromaDB（默认，分片存储，6807 条 ATC 知识已灌入 `data/vector_db`）
- FAISS（可选）
- NumPy（可选，纯暴力检索）

## Deep Agents 动态运行时（可选）

项目默认保留原有固定流程。安装新依赖后，可在 `.env` 中设置：

```bash
AGENT_RUNTIME=deepagents
AGENT_MAX_REFINEMENT_ROUNDS=1
AGENT_REFINEMENT_CONFIDENCE_THRESHOLD=0.80
AGENT_ENABLE_SUBAGENTS=false
```

启用后，单药映射会根据证据缺口动态调用既有的 PubChem、Tavily、WHO ATC 与向量库工具；低置信度或无编码结果会在预算内再推理一次。领域规则位于 `skills/`，只读执行轨迹会记录到 `data/learning/episodes.jsonl`，用于离线评测和人工审核，不会自动修改业务代码或 Skill。

## 具体思路如下

                 Excel
                  │
                  ▼
          读取药物名称
                  │
                  ▼
        Drug Agent（主智能体）
                  │
      ┌───────────┴───────────┐
      │                       │
      ▼                       ▼
    ① 名称标准化              文件缓存检查
    (Translation + LLM)         (避免重复计算)
        │
        ▼
    得到：
    • 英文通用名
    • 中文通用名
    • 药物类别
    • Alias
        │
        ▼
    ② 多源知识检索（Retriever）
        │
    ┌───────────────────────────────┐
    │          │          │         │
    ▼          ▼          ▼
    PubChem    Tavily   VectorDB(RAG)
    │          │          │
    └──────────┴──────────┴─────────┘
                        │
                        ▼
            汇总所有上下文(Context)
                        │
                        ▼
    ③ LLM推理ATC编码
                        │
        Prompt + Retrieval Context
                        │
                        ▼
                LLM生成JSON结果
                        │
                        ▼
            输出：
            • ATC编码
            • 各级ATC(Level1~5)
            • 推理依据
            • 候选ATC
            • Confidence
                        │
                        ▼
    ④ Reflection Agent
                        │
        判断Confidence是否足够高
                        │
            ┌─────────┴─────────┐
            │                   │
        ≥0.90                  <0.90
            │                   │
            ▼                   ▼
        直接通过        再检索WHO+PubChem
                            │
                            ▼
                    再次调用LLM审核
                            │
                            ▼
                    输出最终ATC编码
                            │
                            ▼
            写入Cache + 输出Excel
