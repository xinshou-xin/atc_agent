"""
百度翻译工具
"""
import hashlib
import random
import time
import requests
from config.settings import BAIDU_TRANSLATE
from utils.logger import setup_logger
from utils.cache import Cache

logger = setup_logger(__name__)


class BaiduTranslator:
    """百度翻译 API 客户端"""

    def __init__(self):
        self.app_id = BAIDU_TRANSLATE["app_id"]
        self.app_key = BAIDU_TRANSLATE["app_key"]
        self.url = BAIDU_TRANSLATE["url"]
        self.from_lang = BAIDU_TRANSLATE["from_lang"]
        self.to_lang = BAIDU_TRANSLATE["to_lang"]
        self.cache = Cache()

    def translate(self, text: str, from_lang: str = None, to_lang: str = None) -> str:
        """
        翻译文本

        Args:
            text: 待翻译文本
            from_lang: 源语言，默认 zh
            to_lang: 目标语言，默认 en

        Returns:
            翻译后的文本
        """
        if not text or not text.strip():
            return text

        from_lang = from_lang or self.from_lang
        to_lang = to_lang or self.to_lang

        # 缓存检查
        cache_key = f"baidu_trans:{from_lang}2{to_lang}:{text}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        if not self.app_id or not self.app_key:
            logger.warning("百度翻译 API 未配置，返回原文")
            return text

        try:
            salt = str(random.randint(32768, 65536))
            sign = self._make_sign(text, salt)

            params = {
                "q": text,
                "from": from_lang,
                "to": to_lang,
                "appid": self.app_id,
                "salt": salt,
                "sign": sign,
            }

            response = requests.post(self.url, params=params, timeout=10)
            response.raise_for_status()
            result = response.json()

            if "trans_result" in result:
                translated = result["trans_result"][0]["dst"]
                self.cache.set(cache_key, translated)
                return translated
            else:
                error_msg = result.get("error_msg", "未知错误")
                logger.error(f"百度翻译错误: {error_msg}")
                return text

        except Exception as e:
            logger.error(f"翻译失败: {e}")
            return text

    def _make_sign(self, text: str, salt: str) -> str:
        """生成签名"""
        sign_str = f"{self.app_id}{text}{salt}{self.app_key}"
        return hashlib.md5(sign_str.encode("utf-8")).hexdigest()

    def translate_batch(self, texts: list, from_lang: str = None, to_lang: str = None) -> list:
        """批量翻译"""
        results = []
        for text in texts:
            results.append(self.translate(text, from_lang, to_lang))
            time.sleep(0.1)  # 避免频率限制
        return results
