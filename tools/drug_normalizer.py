"""
药物名称标准化工具：用 LLM 将药物名解析为英文常用名 + WHO INN 通用名 + 别名。

替代原百度翻译：百度翻译只能给出英文常用名（如 阿司匹林→Aspirin），
无法给出 WHO 官方 INN（如 acetylsalicylic acid）；LLM 具备医学知识，
可一次输出 translated / inn / aliases / drug_class。

调用 LLM 使用项目已配置的 LLM_CONFIG（DeepSeek），结果按输入名缓存。
"""
import json

from config.settings import LLM_CONFIG
from utils.cache import Cache
from utils.logger import setup_logger

logger = setup_logger(__name__)

# 标准化提示词：强约束 inn 不许编造
_NORMALIZE_SYSTEM_PROMPT = """\
你是药物名称标准化专家。给定一个药物名称（可能是中文名、英文常用名、商品名或缩写），\
输出其标准化信息。

要求：
1. translated: 英文常用名（中文输入时给出对应的英文名；英文输入保持）
2. inn: WHO 国际非专利名（INN）通用名，例如 阿司匹林 → acetylsalicylic acid；\
若是复方制剂或无法确认，必须留空字符串
3. aliases: 常见别名/商品名列表（尽量包含输入的原始名称），没有则空数组
4. drug_class: 药物类别（英文，如 NSAID / salicylate），不确定留空字符串

注意：
- inn 必须是权威且有依据的官方通用名，不确定就留空，严禁编造
- 只输出一个 JSON 对象，不要输出任何其他文字
"""


class DrugNameNormalizer:
    """LLM 药物名标准化客户端"""

    def __init__(self):
        from langchain_openai import ChatOpenAI

        self.cache = Cache()
        self.llm = ChatOpenAI(
            model=LLM_CONFIG["model"],
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"],
            temperature=LLM_CONFIG["temperature"],
            max_tokens=LLM_CONFIG["max_tokens"],
            timeout=LLM_CONFIG["timeout"],
        )

    def normalize(self, drug_name: str) -> str:
        """
        标准化药物名，返回结构化 JSON 字符串（供 Agent 直接使用）

        Returns:
            成功: {"source": "llm_normalize", "ok": true,
                   "data": {"translated": "...", "inn": "...",
                            "aliases": [...], "drug_class": "..."}}
            失败: {"source": "llm_normalize", "ok": false,
                   "error": "...", "data": {"translated": 原输入, ...}}
        """
        name = (drug_name or "").strip()
        if not name:
            return json.dumps({
                "source": "llm_normalize",
                "ok": False,
                "error": "empty input",
                "data": {"translated": "", "inn": "", "aliases": [], "drug_class": ""},
            }, ensure_ascii=False)

        cache_key = f"normalize:{name}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        fallback = {
            "translated": name,
            "inn": "",
            "aliases": [],
            "drug_class": "",
        }

        try:
            content = self.llm.invoke(
                _NORMALIZE_SYSTEM_PROMPT + f"\n输入药物名: {name}"
            ).content
            data = self._parse_json(content)
            if not isinstance(data, dict):
                logger.warning(f"LLM 标准化输出非对象，回退原输入: {name}")
                data = fallback

            # 字段兜底，确保结构完整
            result = {
                "translated": str(data.get("translated") or name),
                "inn": str(data.get("inn") or "").strip(),
                "aliases": [str(a) for a in (data.get("aliases") or [])],
                "drug_class": str(data.get("drug_class") or ""),
            }
            payload = json.dumps({
                "source": "llm_normalize",
                "ok": True,
                "data": result,
            }, ensure_ascii=False)
            self.cache.set(cache_key, payload)
            return payload

        except Exception as e:
            logger.error(f"LLM 标准化失败: {e}")
            return json.dumps({
                "source": "llm_normalize",
                "ok": False,
                "error": str(e),
                "data": fallback,
            }, ensure_ascii=False)

    @staticmethod
    def _parse_json(content: str) -> dict:
        """提取 LLM 输出中的 JSON 对象（兼容 ```json``` 代码块）"""
        content = (content or "").strip()
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, json.JSONDecodeError):
            start, end = content.find("{"), content.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(content[start:end])
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    pass
        return {}
