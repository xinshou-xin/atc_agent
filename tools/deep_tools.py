"""将既有药物服务封装成 Deep Agents 可调用的只读工具。"""
import json
from typing import Any, Callable, Dict, List

from utils.logger import setup_logger

logger = setup_logger(__name__)

# 本地 RAG 向量库惰性单例：避免 make_atc_tools 多次调用时重复加载
# bge-m3 模型（内存/加载开销大）。
_vector_db = None


def _get_vector_db():
    global _vector_db
    if _vector_db is None:
        try:
            from retriever.vector_db import VectorDB
            _vector_db = VectorDB()
            logger.info(f"本地向量库就绪，共 {_vector_db.count()} 条知识")
        except Exception as error:
            _vector_db = None
            logger.warning(f"本地向量库不可用: {error}")
    return _vector_db


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _safe_call(source: str, operation: Callable[[], Any]) -> str:
    """将外部服务错误转为结构化工具结果，交给 Agent 决定是否换策略。"""
    try:
        result = operation()
        return _json({"source": source, "ok": True, "data": result})
    except Exception as error:
        return _json({"source": source, "ok": False, "error": str(error)})


def make_atc_tools() -> List[Any]:
    """创建只读 Tool 集合。所有业务实现继续复用项目原有客户端。"""
    try:
        from langchain_core.tools import tool
    except ImportError as error:
        raise RuntimeError(
            "未安装 Deep Agents 依赖。请先执行 pip install -r requirements.txt"
        ) from error

    from tools.atc_search import ATCSearcher
    from tools.pubchem import PubChemClient
    from tools.tavily import TavilyClient
    from tools.translator import BaiduTranslator

    translator = BaiduTranslator()
    pubchem = PubChemClient()
    tavily = TavilyClient()
    atc_searcher = ATCSearcher()

    @tool
    def translate_drug_name(drug_name: str) -> str:
        """将中文药物名称翻译为英文检索名；英文输入会原样返回。"""
        if not any("\u4e00" <= char <= "\u9fff" for char in drug_name):
            return _json({"source": "input", "ok": True, "data": drug_name})
        return _safe_call("baidu_translate", lambda: translator.translate(drug_name))

    @tool
    def search_pubchem(drug_name: str) -> str:
        """从 PubChem 查询药物性质、别名和可用的 ATC 相关信息。"""
        return _safe_call("pubchem", lambda: pubchem.get_drug_info(drug_name))

    @tool
    def search_tavily(drug_name: str) -> str:
        """从 Tavily 网络搜索补充药物概述；仅作为低优先级辅助证据。"""
        return _safe_call("tavily", lambda: tavily.get_drug_info(drug_name))

    @tool
    def search_who_atc(drug_name: str) -> str:
        """从 WHO ATC 索引按药物名称查候选编码；这是 ATC 判定的高优先级证据。"""
        return _safe_call("who_atc", lambda: atc_searcher.search_drug(drug_name))

    @tool
    def validate_who_atc(atc_code: str) -> str:
        """从 WHO ATC 索引核验一个候选编码及其官方层级。"""
        return _safe_call("who_atc", lambda: atc_searcher.lookup_atc(atc_code))

    @tool
    def search_local_atc_knowledge(query: str, top_k: int = 5) -> str:
        """检索项目本地 ATC 向量知识库；结果仅作为辅助证据，不能替代 WHO 核验。"""
        vector_db = _get_vector_db()
        if vector_db is None:
            return _json({
                "source": "vector_db",
                "ok": False,
                "error": "local vector db unavailable (init failed)",
            })
        try:
            results = vector_db.search(query, top_k=top_k)
            return _json({"source": "vector_db", "ok": True, "data": results})
        except Exception as error:
            return _json({"source": "vector_db", "ok": False, "error": str(error)})

    return [
        translate_drug_name,
        search_pubchem,
        search_tavily,
        search_who_atc,
        validate_who_atc,
        search_local_atc_knowledge,
    ]
