"""
向量数据库
支持：
1. ChromaDB
2. FAISS
3. numpy（纯 NumPy 暴力检索，避开 Windows 下 torch 与 chromadb/faiss 底层库同进程段错误）

Embedding统一由EmbeddingClient负责
"""

import json
import os

import numpy as np

from config.settings import VECTOR_DB
from utils.logger import setup_logger
from retriever.embedding import EmbeddingClient

logger = setup_logger(__name__)

# Chroma 单集合上限：本机 Windows + Python 3.14 下 chromadb compactor 对
# 超过该规模的 HNSW 索引无法落盘（跨进程打开报 "Error loading hnsw index"），
# 因此按该大小分片存储，检索时合并各分片结果。
CHROMA_SHARD_SIZE = 400
# 剩余条数低于该值时并入当前分片，避免产生极小分片（本机 chromadb 对
# 极小集合的 HNSW 索引同样不落盘，跨进程检索报 "Nothing found on disk"）。
CHROMA_MERGE_THRESHOLD = 100


class VectorDB:
    """向量数据库统一接口"""

    def __init__(self):

        self.provider = VECTOR_DB["provider"]

        self.persist_dir = VECTOR_DB["persist_dir"]

        self.collection_name = VECTOR_DB["collection_name"]

        self.embedding_client = EmbeddingClient()

        self._client = None

        self._collection = None

        # chroma provider 分片集合列表（shard_0、shard_1、...）
        self._shards = []

        # numpy provider 状态（向量已由 EmbeddingClient 归一化）
        self._vectors = None
        self._ids = []
        self._documents = []
        self._metadatas = []

        self._init_db()

    def _init_db(self):
        """初始化向量数据库"""

        os.makedirs(
            self.persist_dir,
            exist_ok=True
        )

        if self.provider == "chroma":

            self._init_chroma()

        elif self.provider == "faiss":

            self._init_faiss()

        elif self.provider == "numpy":

            self._init_numpy()

        else:

            raise ValueError(
                f"不支持的向量数据库: {self.provider}"
            )

    def _shard_name(self, index: int) -> str:
        return f"{self.collection_name}_shard_{index}"

    def _init_chroma(self):
        """初始化 ChromaDB（按分片加载，避免单集合 HNSW 落盘失败）"""

        try:

            import chromadb

            self._client = chromadb.PersistentClient(
                path=self.persist_dir
            )

            # 加载已有分片（按名字排序），没有则创建 0 号分片
            self._shards = sorted(
                [
                    c for c in self._client.list_collections()
                    if c.name.startswith(self.collection_name + "_shard_")
                ],
                key=lambda c: int(c.name.rsplit("_", 1)[-1]),
            )
            if not self._shards:
                self._shards = [
                    self._client.get_or_create_collection(
                        name=self._shard_name(0)
                    )
                ]

            logger.info(
                f"ChromaDB 初始化成功，{len(self._shards)} 个分片"
            )

        except ImportError:

            logger.warning(
                "chromadb 未安装"
            )

        except Exception as e:

            logger.error(
                f"ChromaDB 初始化失败: {e}"
            )

    def _init_faiss(self):
        """初始化 FAISS"""

        try:

            import faiss

            self._np = np

            dimension = 1024

            index_path = os.path.join(
                self.persist_dir,
                "faiss.index"
            )

            if os.path.exists(index_path):

                self._client = (
                    faiss.read_index(index_path)
                )

            else:

                self._client = (
                    faiss.IndexFlatIP(
                        dimension
                    )
                )

            logger.info(
                "FAISS 初始化成功"
            )

        except ImportError:

            logger.warning(
                "faiss 未安装"
            )

        except Exception as e:

            logger.error(
                f"FAISS 初始化失败: {e}"
            )

    # ============================================================
    # NumPy provider：纯文件存储，不加载任何 chromadb/faiss 依赖
    # ============================================================
    def _numpy_dir(self) -> str:
        return os.path.join(self.persist_dir, "numpy")

    def _init_numpy(self):
        """加载纯 NumPy 存储（vectors.npy + 三个 JSON）。"""
        os.makedirs(self._numpy_dir(), exist_ok=True)
        vec_path = os.path.join(self._numpy_dir(), "vectors.npy")
        if os.path.exists(vec_path):
            self._vectors = np.load(vec_path)
            self._ids = self._load_json("ids.json")
            self._documents = self._load_json("documents.json")
            self._metadatas = self._load_json("metadatas.json")
            logger.info(f"NumPy 向量库加载成功，共 {len(self._ids)} 条")
        else:
            logger.info("NumPy 向量库为空，等待灌库")

    def _load_json(self, name: str) -> list:
        path = os.path.join(self._numpy_dir(), name)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _numpy_save(self):
        """全量落盘（向量矩阵 + 三个 JSON）。"""
        base = self._numpy_dir()
        np.save(os.path.join(base, "vectors.npy"), self._vectors)
        for name, data in (
            ("ids.json", self._ids),
            ("documents.json", self._documents),
            ("metadatas.json", self._metadatas),
        ):
            with open(os.path.join(base, name), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

    def add(
        self,
        texts: list,
        ids: list = None,
        metadatas: list = None
    ):
        """
        添加文本到向量库

        Args:
            texts: 文本列表
            ids: ID列表
            metadatas: 元数据列表
        """

        if not texts:

            return

        ids = ids or [
            str(i)
            for i in range(len(texts))
        ]

        metadatas = metadatas or [
            {}
            for _ in texts
        ]

        try:

            # ======================================
            # 1. 生成 Embedding
            # ======================================

            embeddings = (
                self.embedding_client.embed_batch(
                    texts
                )
            )

            if not embeddings:

                logger.warning(
                    "Embedding生成失败，跳过添加"
                )

                return

            # ======================================
            # 2. Chroma
            # ======================================

            if (
                self.provider == "chroma"
                and self._shards
            ):

                # 按分片容量分发，每片不超过 CHROMA_SHARD_SIZE 条；
                # 剩余不足 CHROMA_MERGE_THRESHOLD 条时并入当前片，避免
                # 产生无法落盘的小尾巴分片。
                idx = 0
                total = len(texts)
                while idx < total:
                    cur = self._shards[-1]
                    remaining = total - idx
                    if cur.count() >= CHROMA_SHARD_SIZE:
                        if remaining < CHROMA_MERGE_THRESHOLD:
                            take = remaining
                        else:
                            self._shards.append(
                                self._client.get_or_create_collection(
                                    name=self._shard_name(len(self._shards))
                                )
                            )
                            take = 0
                    else:
                        take = min(
                            CHROMA_SHARD_SIZE - cur.count(),
                            remaining,
                        )
                    if take:
                        cur = self._shards[-1]
                        cur.add(
                            documents=texts[idx:idx + take],
                            embeddings=embeddings[idx:idx + take],
                            ids=ids[idx:idx + take],
                            metadatas=metadatas[idx:idx + take],
                        )
                        idx += take

            # ======================================
            # 3. FAISS
            # ======================================

            elif (
                self.provider == "faiss"
                and self._client
            ):

                vectors = np.array(
                    embeddings,
                    dtype="float32"
                )

                self._client.add(
                    vectors
                )

                index_path = os.path.join(
                    self.persist_dir,
                    "faiss.index"
                )

                import faiss

                faiss.write_index(
                    self._client,
                    index_path
                )

            # ======================================
            # 4. NumPy：追加保存（向量已归一化）
            # ======================================

            elif self.provider == "numpy":

                new_vectors = np.asarray(
                    embeddings,
                    dtype="float32"
                )

                if (
                    self._vectors is None
                    or self._vectors.shape[0] == 0
                ):

                    self._vectors = new_vectors

                else:

                    self._vectors = np.vstack(
                        [self._vectors, new_vectors]
                    )

                self._ids.extend(ids)
                self._documents.extend(texts)
                self._metadatas.extend(metadatas)
                self._numpy_save()

            logger.info(
                f"添加 {len(texts)} 条文本到向量库"
            )

        except Exception as e:

            logger.error(
                f"向量添加失败: {e}"
            )

    def search(
        self,
        query: str,
        top_k: int = 20
    ) -> list:
        """
        向量检索

        Args:
            query: 查询文本
            top_k: 初始召回数量

        Returns:
            [
                {
                    "text": "...",
                    "score": 0.92,
                    "metadata": {}
                }
            ]
        """

        if not query:

            return []

        try:

            # ======================================
            # 1. Query Embedding
            # ======================================

            query_embedding = (
                self.embedding_client.embed(
                    query
                )
            )

            if not query_embedding:

                return []

            # ======================================
            # 2. Chroma
            # ======================================

            if (
                self.provider == "chroma"
                and self._shards
            ):

                # 各分片独立查询后合并，chroma 的 distance 越小越相关。
                # 本机 chromadb 偶发 "Nothing found on disk"（HNSW 段读取
                # 瞬时失败），此处对失败分片跳过并整体重试，保证可用性。
                import time

                output = []
                for attempt in range(3):
                    output = []
                    failed = 0
                    for shard in self._shards:
                        try:
                            count = shard.count()
                            if count == 0:
                                continue
                            n = min(top_k, count)
                            results = shard.query(
                                query_embeddings=[query_embedding],
                                n_results=n,
                            )
                            for doc, distance, metadata in zip(
                                results["documents"][0],
                                results["distances"][0],
                                results["metadatas"][0],
                            ):
                                output.append({
                                    "text": doc,
                                    "score": distance,
                                    "metadata": metadata,
                                })
                        except Exception as e:
                            failed += 1
                            logger.warning(
                                f"分片 {shard.name} 检索失败，跳过: {e}"
                            )
                    if output or attempt == 2:
                        break
                    time.sleep(2)

                output.sort(key=lambda x: x["score"])
                return output[:top_k]

            # ======================================
            # 3. FAISS
            # ======================================

            elif (
                self.provider == "faiss"
                and self._client
            ):

                vector = np.array(
                    [query_embedding],
                    dtype="float32"
                )

                scores, indices = (
                    self._client.search(
                        vector,
                        top_k
                    )
                )

                return [
                    {
                        "text": "",
                        "score": float(
                            scores[0][i]
                        ),
                        "metadata": {
                            "index": int(
                                indices[0][i]
                            )
                        }
                    }
                    for i in range(
                        len(indices[0])
                    )
                    if indices[0][i] >= 0
                ]

            # ======================================
            # 4. NumPy：点积即余弦相似度（向量已归一化）
            # ======================================

            elif (
                self.provider == "numpy"
                and self._vectors is not None
                and self._vectors.shape[0] > 0
            ):

                qv = np.asarray(
                    query_embedding,
                    dtype="float32"
                )

                scores = self._vectors @ qv

                top_k = min(
                    top_k,
                    len(self._ids)
                )

                idx = np.argsort(-scores)[:top_k]

                return [
                    {
                        "text": self._documents[int(i)],
                        "score": float(scores[int(i)]),
                        "metadata": self._metadatas[int(i)],
                    }
                    for i in idx
                ]

        except Exception as e:

            logger.error(
                f"向量检索失败: {e}"
            )

        return []

    def count(self) -> int:

        if self.provider == "chroma" and self._shards:

            return sum(s.count() for s in self._shards)

        if self.provider == "faiss" and self._client:

            return self._client.ntotal

        if self.provider == "numpy":

            return (
                0
                if self._vectors is None
                else self._vectors.shape[0]
            )

        return 0
