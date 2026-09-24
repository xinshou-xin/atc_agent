"""
全局配置：API Key、URL、参数等
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 项目根目录（config/ 的上一级），用于锚定相对路径，
# 避免在不同工作目录下运行时把文件写到错误位置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==================== LLM 配置 ====================
LLM_CONFIG = {
    "provider": os.getenv("LLM_PROVIDER", "deepseek"),  # deepseek / qwen / openai
    "api_key": os.getenv("LLM_API_KEY", ""),
    "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    "model": os.getenv("LLM_MODEL", "deepseek-chat"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
    "timeout": int(os.getenv("LLM_TIMEOUT", "60")),
}

# ==================== Agent 运行时 ====================
# legacy: 保持原有固定流程；deepagents: 启用动态规划、Skills 与工具调用循环。
# 为了兼容现有环境，默认仍使用 legacy；在 .env 中设置 AGENT_RUNTIME=deepagents 后启用新流程。
AGENT_CONFIG = {
    "runtime": os.getenv("AGENT_RUNTIME", "legacy").strip().lower(),
    "max_refinement_rounds": int(os.getenv("AGENT_MAX_REFINEMENT_ROUNDS", "1")),
    "refinement_confidence_threshold": float(
        os.getenv("AGENT_REFINEMENT_CONFIDENCE_THRESHOLD", "0.80")
    ),
    # 默认不启用通用子 Agent，避免每个药物都产生额外的模型调用。
    "enable_subagents": os.getenv("AGENT_ENABLE_SUBAGENTS", "false").lower() == "true",
}

# 递归改进产生的是可审计的学习记录，而非自动修改 Prompt、Skill 或业务代码。
LEARNING_CONFIG = {
    "enabled": os.getenv("LEARNING_ENABLED", "true").lower() == "true",
    "dir": os.path.join(PROJECT_ROOT, "data", "learning"),
}

# ==================== PubChem ====================
PUBCHEM = {
    "base_url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug",
    # PUG View 接口：ATC 编码需从该接口的 "ATC Code" 节点获取（PUG REST 无 /atc/ 端点）
    "pug_view_url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view",
    "timeout": 30,
    "retry": 3,
}

# ==================== Tavily 搜索 ====================
TAVILY = {
    "api_key": os.getenv("TAVILY_API_KEY", ""),
    "base_url": "https://api.tavily.com/search",
    "timeout": 30,
    "max_results": 5,
}

# ==================== ATC 搜索 ====================
ATC_SEARCH = {
    "who_url": "https://atcddd.fhi.no/atc_ddd_index/",
    "timeout": 30,
}

# ==================== 向量数据库 ====================
VECTOR_DB = {
    "provider": os.getenv("VECTOR_DB_PROVIDER", "chroma"),  # chroma / faiss
    "persist_dir": os.path.join(PROJECT_ROOT, "data", "vector_db"),
    "collection_name": "drug_atc_knowledge",
}

# ==================== Embedding ====================
EMBEDDING = {
    "provider": os.getenv("EMBEDDING_PROVIDER", "local"),  # local / openai / api

    # local
    "model_path": os.getenv("EMBEDDING_MODEL_PATH",""),
    # api（openai 兼容）
    "api_key": os.getenv("EMBEDDING_API_KEY", ""),
    "base_url": os.getenv("EMBEDDING_BASE_URL","https://api.openai.com/v1"),
    "model": os.getenv("EMBEDDING_MODEL","text-embedding-3-small"),
    "dimension": 1024,
    # 自建 BGE-M3 服务（FastAPI：POST /embedding）
    "api_url": os.getenv("EMBEDDING_API_URL", ""),
    "api_token": os.getenv("EMBEDDING_API_TOKEN", ""),
}


# ==================== Reranker ====================
RERANKER = {
    "provider": os.getenv("RERANKER_PROVIDER", "local"),  # local / cohere / api
    "model_path": os.getenv("RERANKER_MODEL_PATH", ""),
    "api_key": os.getenv("RERANKER_API_KEY", ""),
    "model": os.getenv("RERANKER_MODEL", "rerank-multilingual-v3.0"),
    "top_n": 5,
    # 多卡机器上建议指定单卡（如 "cuda:0"），避免多卡初始化慢/死锁；
    # 留空则走默认行为（GPU 可用时自动多卡）
    "devices": os.getenv("RERANKER_DEVICES", "cuda:0"),
    # 自建 BGE-Reranker 服务（FastAPI：POST /rank）
    "api_url": os.getenv("RERANKER_API_URL", ""),
    "api_token": os.getenv("RERANKER_API_TOKEN", ""),
}

# ==================== 缓存 ====================
CACHE = {
    "enabled": True,
    "dir": os.path.join(PROJECT_ROOT, "data", "cache"),
    "ttl_days": 30,
}

# ==================== 日志 ====================
LOG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "file": os.path.join(PROJECT_ROOT, "data", "logs", "app.log"),
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}
