"""
Embedding 工具
支持：
1. 本地 BGE-M3
2. OpenAI 兼容 API
3. 自建 BGE-M3 API（FastAPI，POST /embedding）
"""

import math

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

        elif self.provider == "api":
            self.api_url = EMBEDDING["api_url"]
            self.api_token = EMBEDDING["api_token"]

            logger.info(f"使用自建 Embedding API: {self.api_url}")

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

            elif self.provider == "api":

                vector = self._embed_api(text)

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

            elif self.provider == "api":

                valid_texts = [
                    text.strip()
                    for text in texts
                    if text and text.strip()
                ]

                if not valid_texts:
                    return []

                return self._embed_api_batch(valid_texts)

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

    def _embed_api(self, text: str) -> list:
        """自建 BGE-M3 API（单条）"""

        vectors = self._embed_api_batch([text])

        return vectors[0] if vectors else []

    def _embed_api_batch(self, texts: list) -> list:
        """自建 BGE-M3 API（批量，一次请求返回全部向量）

        服务器契约（FastAPI）:
            POST {api_url}/embedding  header: token
            body: {"texts": ["..."]}
            resp: {"embeddings": [[...]], "count": n}

        服务器返回的是未归一化的 CLS 向量，这里做 L2 归一化，
        与 local provider（normalize_embeddings=True）行为保持一致，
        保证 numpy provider 的点积即余弦相似度。
        """

        if not self.api_url:

            logger.warning(
                "Embedding API 地址未配置（EMBEDDING_API_URL）"
            )

            return []

        headers = {"Content-Type": "application/json"}

        if self.api_token:

            headers["token"] = self.api_token

        response = requests.post(
            self.api_url.rstrip("/") + "/embedding",
            headers=headers,
            json={"texts": texts},
            timeout=60,
        )

        response.raise_for_status()

        result = response.json()

        embeddings = result.get("embeddings") or []

        return [
            self._normalize(v)
            for v in embeddings
        ]

    @staticmethod
    def _normalize(vector: list) -> list:
        """L2 归一化向量"""

        norm = math.sqrt(
            sum(x * x for x in vector)
        )

        if norm == 0:

            return vector

        return [x / norm for x in vector]

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