"""
Reranker 重排序工具

支持：
1. 本地 BGE-Reranker-v2-m3
2. Cohere Rerank API
"""

import requests

from config.settings import RERANKER
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Reranker:
    """重排序器"""

    def __init__(self):

        self.provider = RERANKER["provider"]

        self.top_n = RERANKER["top_n"]

        if self.provider == "local":

            from FlagEmbedding import FlagReranker

            self.model_path = (
                RERANKER["model_path"]
            )

            logger.info(
                f"加载本地 Reranker: "
                f"{self.model_path}"
            )

            # 设备配置：多卡机器默认指定单卡，避免多卡初始化慢/死锁
            devices = RERANKER.get("devices") or None

            self.model = FlagReranker(
                self.model_path,
                use_fp16=True,
                devices=devices,
            )

            logger.info(
                "BGE Reranker 加载完成"
            )

        elif self.provider == "cohere":

            self.api_key = (
                RERANKER["api_key"]
            )

            self.model_name = (
                RERANKER["model"]
            )

        else:

            raise ValueError(
                f"不支持的 reranker provider: "
                f"{self.provider}"
            )

    def rerank(
        self,
        query: str,
        documents: list,
        top_n: int = None
    ) -> list:
        """
        对检索结果进行重排序

        Args:
            query:
                用户查询

            documents:
                文档列表

            top_n:
                返回Top N

        Returns:
            [
                {
                    "text": "...",
                    "score": 0.95,
                    "index": 2
                }
            ]
        """

        if not query or not documents:

            return []

        top_n = top_n or self.top_n

        try:

            if self.provider == "local":

                return self._rerank_bge(
                    query,
                    documents,
                    top_n
                )

            elif self.provider == "cohere":

                return self._rerank_cohere(
                    query,
                    documents,
                    top_n
                )

        except Exception as e:

            logger.error(
                f"Rerank失败: {e}"
            )

            return [
                {
                    "text": doc,
                    "score": 0.0,
                    "index": i
                }
                for i, doc in enumerate(
                    documents[:top_n]
                )
            ]

        return []

    def _rerank_bge(
        self,
        query: str,
        documents: list,
        top_n: int
    ) -> list:
        """
        本地 BGE-Reranker-v2-m3
        """

        pairs = [
            [
                query,
                doc
            ]
            for doc in documents
        ]

        scores = (
            self.model.compute_score(
                pairs
            )
        )

        # 单条情况下返回float
        if not isinstance(
            scores,
            list
        ):

            scores = [scores]

        results = []

        for i, score in enumerate(
            scores
        ):

            results.append(
                {
                    "text": documents[i],
                    "score": float(score),
                    "index": i
                }
            )

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:top_n]

    def _rerank_cohere(
        self,
        query: str,
        documents: list,
        top_n: int
    ) -> list:
        """Cohere Rerank API"""

        url = (
            "https://api.cohere.ai/v1/rerank"
        )

        headers = {
            "Authorization":
                f"Bearer {self.api_key}",

            "Content-Type":
                "application/json"
        }

        data = {
            "model":
                self.model_name,

            "query":
                query,

            "documents":
                documents,

            "top_n":
                top_n
        }

        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=30
        )

        response.raise_for_status()

        result = response.json()

        return [
            {
                "text":
                    documents[r["index"]],

                "score":
                    r["relevance_score"],

                "index":
                    r["index"]
            }

            for r in result.get(
                "results",
                []
            )
        ]