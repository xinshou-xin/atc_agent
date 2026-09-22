"""
PubChem 工具
"""
import re
import time
import requests
from config.settings import PUBCHEM
from utils.logger import setup_logger
from utils.cache import Cache

logger = setup_logger(__name__)

# ATC 编码格式：1 字母 + 2 数字 + 1~2 字母 + 2 数字，共 7 位，如 B01AC06 / N02BA01。
# 用 findall 提取子串，可覆盖 "B01AC06; N02BA01" 分号拼接和 "N02BA01 - 名称" 层级条目；
# \b 单词边界用于排除 ATCvet 兽医用药 Q 系列（如 QB01AC06），避免其被误提取为 B01AC06。
_ATC_CODE_RE = re.compile(r"\b[A-Z]\d{2}[A-Z]{1,2}\d{2}\b")


class PubChemClient:
    """PubChem API 客户端"""

    def __init__(self):
        self.base_url = PUBCHEM["base_url"]
        self.pug_view_url = PUBCHEM["pug_view_url"]
        self.timeout = PUBCHEM["timeout"]
        self.retry = PUBCHEM["retry"]
        self.cache = Cache()

    def get_drug_info(self, drug_name: str) -> str:
        cache_key = f"pubchem_info:{drug_name}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        try:
            cid = self._get_cid(drug_name)
            if not cid:
                logger.warning(f"PubChem 未找到: {drug_name}")
                return ""
            properties = self._get_properties(cid)
            atc_codes = self._get_atc_codes(cid)
            info_parts = [f"药物名称: {drug_name}", f"PubChem CID: {cid}"]
            if properties:
                info_parts.append(f"分子式: {properties.get('MolecularFormula', 'N/A')}")
                info_parts.append(f"分子量: {properties.get('MolecularWeight', 'N/A')}")
                info_parts.append(f"IUPAC名称: {properties.get('IUPACName', 'N/A')}")
            if atc_codes:
                info_parts.append(f"ATC 编码: {', '.join(atc_codes)}")
                # 顺带缓存 ATC，供 get_atc_code 复用，避免同一药物重复请求 PUG View（易被限流 503）
                self.cache.set(f"pubchem_atc:{drug_name}", atc_codes)
            result = "\n".join(info_parts)
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"PubChem 获取信息失败: {e}")
            return ""

    def get_atc_code(self, drug_name: str) -> str:
        try:
            cached = self.cache.get(f"pubchem_atc:{drug_name}")
            if cached:
                return cached[0] if cached else ""
            cid = self._get_cid(drug_name)
            if cid:
                atc_codes = self._get_atc_codes(cid)
                if atc_codes:
                    self.cache.set(f"pubchem_atc:{drug_name}", atc_codes)
                return atc_codes[0] if atc_codes else ""
        except Exception as e:
            logger.error(f"PubChem ATC 查询失败: {e}")
        return ""

    def _get_cid(self, drug_name: str) -> str:
        url = f"{self.base_url}/compound/name/{drug_name}/cids/JSON"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        return str(cids[0]) if cids else ""

    def _get_properties(self, cid: str) -> dict:
        props = "MolecularFormula,MolecularWeight,IUPACName"
        url = f"{self.base_url}/compound/cid/{cid}/property/{props}/JSON"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        properties = data.get("PropertyTable", {}).get("Properties", [])
        return properties[0] if properties else {}

    def _get_atc_codes(self, cid: str) -> list:
        """从 PUG View 数据中提取 ATC 编码。

        PUG REST 没有 /atc/ 端点（/compound/cid/{cid}/atc/JSON 返回 400 BadRequest），
        ATC 编码位于 PUG View 返回的 Record.Section 树中 TOCHeading == "ATC Code"
        的节点下，表现为三种形态：
          - 纯编码项：Value.StringWithMarkup[0].String == "B01AC06"
          - 层级条目：末级 String == "B01AC06 - Acetylsalicylic acid"
          - 拼接项：String == "B01AC06; N02BA01"
        统一按 ATC 编码格式提取子串；\b 边界排除 ATCvet 兽医 Q 系列。

        PUG View 是大响应接口（约 1.8MB），PubChem 繁忙时会返回 503 ServerBusy，
        属临时状态，按配置的 retry 次数做指数退避重试（1s/2s/4s）。
        """
        url = f"{self.pug_view_url}/data/compound/{cid}/JSON"
        for attempt in range(max(1, self.retry)):
            try:
                response = requests.get(url, timeout=self.timeout)
                if response.status_code == 503:
                    wait = 2 ** attempt
                    logger.warning(f"PubChem PUG View 繁忙(503)，{wait}s 后重试 ({attempt + 1}/{self.retry})")
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                data = response.json()
                codes = set()
                for section in self._iter_sections(data.get("Record", {})):
                    if section.get("TOCHeading") != "ATC Code":
                        continue
                    for info in section.get("Information", []):
                        for markup in info.get("Value", {}).get("StringWithMarkup", []):
                            text = markup.get("String", "")
                            codes.update(_ATC_CODE_RE.findall(text))
                return sorted(codes)
            except Exception as e:
                logger.error(f"PubChem ATC 获取失败: {e}")
                return []
        logger.error(f"PubChem ATC 获取失败: PUG View 繁忙，重试 {self.retry} 次后仍失败")
        return []

    @staticmethod
    def _iter_sections(node):
        """递归遍历 PUG View 的 Record.Section 树，产出所有含 TOCHeading 的节点。"""
        if isinstance(node, dict):
            for sub in node.get("Section", []):
                yield sub
                yield from PubChemClient._iter_sections(sub)
