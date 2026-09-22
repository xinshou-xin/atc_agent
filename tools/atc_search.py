"""
WHO ATC 官网搜索工具
官网: https://atcddd.fhi.no/atc_ddd_index/
- 按名称搜索: GET ?name=xxx（仅支持 INN 通用名，前缀匹配，最少 3 个字母）
- 按编码查详情: GET ?code=N02BA01（返回上级层级链、物质名、DDD）
"""
import re
import time
import requests
from bs4 import BeautifulSoup
from config.settings import ATC_SEARCH
from utils.logger import setup_logger
from utils.cache import Cache

logger = setup_logger(__name__)

# 匹配链接里的 ATC 编码，如 ./?code=N02BA01&showdescription=no
CODE_RE = re.compile(r"[?&]code=([A-Z][A-Z0-9]{0,6})")


class ATCSearcher:
    """WHO ATC 编码搜索工具"""

    def __init__(self):
        self.base_url = ATC_SEARCH["who_url"]
        self.timeout = ATC_SEARCH["timeout"]
        self.cache = Cache()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; drug-atc-research/1.0)"
        })

    # ------------------------------------------------------------------
    # 按药物名搜索（供 DrugAgent 检索阶段使用）
    # ------------------------------------------------------------------
    def search_drug(self, drug_name: str) -> list:
        """
        按 INN 通用名搜索 WHO 官网

        Args:
            drug_name: 药物英文名（应为 INN 通用名，商品名搜不到）

        Returns:
            [{"code": "N02BA01", "name": "acetylsalicylic acid", "level": 5}, ...]
            未找到返回 []
        """
        name = (drug_name or "").strip().lower()
        if len(name) < 3:  # 官网要求最少 3 个字母
            return []

        cache_key = f"atc_search:{name}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        results = []
        try:
            resp = self.session.get(
                self.base_url, params={"name": name}, timeout=self.timeout
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            if "No match found" in soup.get_text():
                self.cache.set(cache_key, [])
                return []

            seen = set()
            for a in soup.find_all("a", href=True):
                m = CODE_RE.search(a["href"])
                if not m:
                    continue
                code = m.group(1)
                level = self._get_atc_level(code)
                if level == 0 or code in seen:
                    continue
                seen.add(code)
                results.append({
                    "code": code,
                    "name": a.get_text(strip=True),
                    "level": level,
                })

            self.cache.set(cache_key, results)
            time.sleep(0.5)  # 对官网保持礼貌
        except Exception as e:
            logger.error(f"ATC 搜索失败: {e}")

        return results

    # ------------------------------------------------------------------
    # 按编码查详情（供 ReflectionAgent 验证编码真实性）
    # ------------------------------------------------------------------
    def lookup_atc(self, atc_code: str) -> dict:
        """
        根据 ATC 编码到官网验证并获取层级信息

        Returns:
            成功: {
                "code": "N02BA01", "valid": True, "level": 5,
                "name": "acetylsalicylic acid",
                "hierarchy": {"N": "NERVOUS SYSTEM", "N02": "ANALGESICS", ...},
            }
            官网查无此码: {"code": ..., "valid": False}
            请求出错: {}
        """
        atc_code = (atc_code or "").strip().upper()
        if self._get_atc_level(atc_code) == 0:
            return {"code": atc_code, "valid": False}

        cache_key = f"atc_lookup:{atc_code}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            resp = self.session.get(
                self.base_url,
                params={"code": atc_code, "showdescription": "no"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            if "No match found" in soup.get_text():
                negative = {"code": atc_code, "valid": False}
                self.cache.set(cache_key, negative)  # 负缓存，避免反复请求
                return negative

            # 解析上级层级链：页面上所有指向"更短编码"的链接
            hierarchy = {}
            substance = ""
            for a in soup.find_all("a", href=True):
                m = CODE_RE.search(a["href"])
                if not m:
                    continue
                code = m.group(1)
                text = a.get_text(strip=True)
                if code == atc_code:
                    # 指向自己的链接 = 物质名（5 级编码的 DDD 表格中）
                    if text and not substance and self._get_atc_level(code) == 5:
                        substance = text
                    continue
                if self._get_atc_level(code) > 0:
                    hierarchy[code] = text

            result = {
                "code": atc_code,
                "valid": True,
                "level": self._get_atc_level(atc_code),
                "name": substance,
                "hierarchy": hierarchy,
            }
            self.cache.set(cache_key, result)
            time.sleep(0.5)
            return result

        except Exception as e:
            logger.error(f"ATC 查询失败: {e}")
            return {}

    @staticmethod
    def _get_atc_level(atc_code: str) -> int:
        """判断 ATC 编码级别"""
        return {1: 1, 3: 2, 4: 3, 5: 4, 7: 5}.get(len(atc_code or ""), 0)