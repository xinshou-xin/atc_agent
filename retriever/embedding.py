"""
Embedding 工具
支持：
1. 本地 BGE-M3
2. OpenAI 兼容 API
"""

import requests
from config.settings import EMBEDDING
from utils.logger import setup_logger
from utils.cache import Cache

logger = setup_logger(__name__)


class EmbeddingClient:
    """Embedding 统一接口"""

    def __init__(self):
        self.provider = EMBEDDING["provider"]
        self.cache = Cache()

        if self.provider == "local":
            from sentence_transformers import SentenceTransformer

            self.model_path = EMBEDDING["model_path"]

            logger.info(f"加载本地 Embedding 模型: {self.model_path}")

            self.model = SentenceTransformer(
                self.model_path
            )

            logger.info("本地 BGE-M3 加载完成")

        elif self.provider in ["openai", "deepseek", "qwen"]:
            self.api_key = EMBEDDING["api_key"]
            self.base_url = EMBEDDING["base_url"]
            self.model_name = EMBEDDING["model"]

        else:
            raise ValueError(
                f"不支持的 embedding provider: {self.provider}"
            )

    def embed(self, text: str) -> list:
        """
        获取单条文本 embedding 向量

        Args:
            text: 输入文本

        Returns:
            embedding 向量列表
        """

        if not text or not text.strip():
            return []

        text = text.strip()

        cache_key = f"embed:{self.provider}:{text}"

        cached = self.cache.get(cache_key)

        if cached:
            return cached

        try:

            if self.provider == "local":

                vector = self._embed_local(text)

            elif self.provider in [
                "openai",
                "deepseek",
                "qwen"
            ]:

                vector = self._embed_openai_compatible(text)

            else:

                logger.warning(
                    f"不支持的 embedding provider: {self.provider}"
                )

                return []

            self.cache.set(
                cache_key,
                vector
            )

            return vector

        except Exception as e:

            logger.error(
                f"Embedding 失败: {e}"
            )

            return []

    def embed_batch(self, texts: list) -> list:
        """
        批量获取 embedding

        Args:
            texts: 文本列表

        Returns:
            embedding 向量列表
        """

        if not texts:
            return []

        try:

            if self.provider == "local":

                valid_texts = [
                    text.strip()
                    for text in texts
                    if text and text.strip()
                ]

                if not valid_texts:
                    return []

                vectors = self.model.encode(
                    valid_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )

                return vectors.tolist()

            else:

                return [
                    self.embed(text)
                    for text in texts
                ]

        except Exception as e:

            logger.error(
                f"批量 Embedding 失败: {e}"
            )

            return []

    def _embed_local(self, text: str) -> list:
        """本地 BGE-M3"""

        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return vector.tolist()

    def _embed_openai_compatible(self, text: str) -> list:
        """OpenAI 兼容 Embedding API"""

        url = f"{self.base_url.rstrip('/')}/embeddings"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model_name,
            "input": text,
        }

        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        return result["data"][0]["embedding"]