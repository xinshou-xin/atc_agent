"""
缓存工具: SQLite 单文件存储

一个 cache.db 文件存储所有缓存键，替代旧的"一个缓存键一个 JSON 文件"实现，
避免缓存条目增多时产生大量小文件。初始化时自动清理过期项。

接口保持: get(key) / set(key, value, ttl_days=None) / clear()
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from config.settings import CACHE
from utils.logger import setup_logger

logger = setup_logger(__name__)


class Cache:
    """SQLite 文件缓存"""

    def __init__(self):
        self.enabled = CACHE["enabled"]
        self.cache_dir = CACHE["dir"]
        self.ttl_days = CACHE["ttl_days"]
        self.db_path = os.path.join(self.cache_dir, "cache.db")

        if self.enabled:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """新建数据库连接（每次操作短连接，避免连接泄漏）"""
        return sqlite3.connect(self.db_path, timeout=10)

    def _init_db(self):
        """建表 + 启动时清理过期项"""
        try:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cache (
                        key        TEXT PRIMARY KEY,
                        value      TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at)"
                )
                conn.execute(
                    "DELETE FROM cache WHERE expires_at < ?",
                    (datetime.now().isoformat(),),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"缓存初始化失败: {e}")

    def get(self, key: str):
        """
        获取缓存

        Args:
            key: 缓存键

        Returns:
            缓存值，未命中或已过期返回 None
        """
        if not self.enabled:
            return None

        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
                ).fetchone()
            finally:
                conn.close()

            if row is None:
                return None

            value_json, expires_at = row
            if datetime.now() > datetime.fromisoformat(expires_at):
                self._delete(key)
                return None

            return json.loads(value_json)

        except Exception as e:
            logger.warning(f"读取缓存失败: {e}")
            return None

    def set(self, key: str, value, ttl_days: int = None):
        """
        设置缓存

        Args:
            key: 缓存键
            value: 缓存值（可 JSON 序列化的任意类型）
            ttl_days: 过期天数，默认用全局配置
        """
        if not self.enabled:
            return

        try:
            ttl = ttl_days if ttl_days is not None else self.ttl_days
            now = datetime.now()
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache (key, value, created_at, expires_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        key,
                        json.dumps(value, ensure_ascii=False),
                        now.isoformat(),
                        (now + timedelta(days=ttl)).isoformat(),
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"写入缓存失败: {e}")

    def clear(self):
        """清空所有缓存"""
        if not self.enabled or not os.path.exists(self.db_path):
            return

        try:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM cache")
                conn.commit()
            finally:
                conn.close()
            logger.info("缓存已清空")
        except Exception as e:
            logger.warning(f"清空缓存失败: {e}")

    def _delete(self, key: str):
        """删除单个缓存键（内部使用）"""
        try:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"删除缓存失败: {e}")
