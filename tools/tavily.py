"""
Tavily 搜索工具
官网: https://tavily.com
- 按药物名称搜索网络，返回网页摘要和 AI 综合回答
"""
import requests
from config.settings import TAVILY
from utils.logger import setup_logger
from utils.cache import Cache

logger = setup_logger(__name__)


class TavilyClient:
    """Tavily 搜索 API 客户端"""

    def __init__(self):
        self.api_key = TAVILY["api_key"]
        self.base_url = TAVILY["base_url"]
        self.timeout = TAVILY["timeout"]
        self.max_results = TAVILY["max_results"]
        self.cache = Cache()

    def get_drug_info(self, drug_name: str) -> str:
        """
        从 Tavily 搜索药物信息

        Args:
            drug_name: 药物名称（英文）

        Returns:
            药物信息摘要文本（网页结果 + 综合回答），失败返回 ""
        """
        if not self.api_key:
            logger.warning("Tavily API Key 未配置（TAVILY_API_KEY）")
            return ""

        cache_key = f"tavily_info:{drug_name}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            payload = {
                "api_key": self.api_key,
                "query": f"{drug_name} drug ATC classification",
                "search_depth": "basic",
                "include_answer": True,
                "max_results": self.max_results,
            }
            response = requests.post(self.base_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            answer = data.get("answer", "")
            results = data.get("results", [])

            info_parts = [f"Tavily 搜索结果: {drug_name}"]
            if answer:
                info_parts.append(f"综合回答: {answer}")
            for i, item in enumerate(results[: self.max_results], 1):
                title = item.get("title", "")
                url = item.get("url", "")
                content = item.get("content", "")
                info_parts.append(f"{i}. {title} ({url})\n{content}")

            result = "\n".join(info_parts)
            self.cache.set(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Tavily 搜索失败: {e}")
            return ""
