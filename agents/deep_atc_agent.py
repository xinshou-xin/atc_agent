"""基于 Deep Agents 的动态单药 ATC 映射器。"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from config.settings import AGENT_CONFIG, LLM_CONFIG, PROJECT_ROOT
from tools.deep_tools import make_atc_tools
from utils.logger import setup_logger

logger = setup_logger(__name__)

ATC_CODE_PATTERN = re.compile(r"^[A-Z]\d{2}[A-Z]{2}\d{2}$")


class DeepATCAgent:
    """让 Agent 按证据缺口动态选择检索工具，并在必要时做有限轮再推理。"""

    def __init__(self):
        self.max_refinement_rounds = max(0, AGENT_CONFIG["max_refinement_rounds"])
        self.refinement_threshold = AGENT_CONFIG["refinement_confidence_threshold"]
        self.agent = self._create_agent()

    def _create_agent(self):
        try:
            from deepagents import create_deep_agent
            from deepagents.backends.filesystem import FilesystemBackend
            from deepagents.middleware import FilesystemMiddleware
            from langchain_openai import ChatOpenAI
        except ImportError as error:
            raise RuntimeError(
                "AGENT_RUNTIME=deepagents 需要安装 deepagents、langchain-openai。"
                "请先执行 pip install -r requirements.txt"
            ) from error

        # 后端根目录仅包含 Skill，不暴露 .env、缓存或项目源码给模型读取。
        skills_root = Path(PROJECT_ROOT) / "skills"
        backend = FilesystemBackend(root_dir=str(skills_root), virtual_mode=True)
        model = ChatOpenAI(
            model=LLM_CONFIG["model"],
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"],
            temperature=LLM_CONFIG["temperature"],
            max_tokens=LLM_CONFIG["max_tokens"],
            timeout=LLM_CONFIG["timeout"],
        )

        system_prompt = """
            你是药物 ATC 映射的主 Agent。你的任务是为一个药物确定 WHO ATC 编码，并生成可审计的 JSON。

            工作规则：
            1. 先判断名称是否需要翻译或存在歧义；需要时调用 translate_drug_name。
            2. 优先查询 search_who_atc，并对准备输出的每个 7 位编码调用 validate_who_atc。
            3. PubChem、Tavily、向量库是辅助证据；Tavily 仅用于补充背景，不能覆盖 WHO 结论。
            4. 证据冲突、无官方命中或名称歧义时，继续检索或降低置信度；禁止凭空编造 ATC 编码。
            5. 每轮只调用解决当前证据缺口所需的工具。若连续检索没有新增有效证据，应停止并标记需要人工核验。
            6. 最终只能输出一个 JSON 对象，格式：
            {
            "normalized": {"generic_name_en": "", "generic_name_cn": "", "aliases": [], "drug_class": ""},
            "atc_result": {
                "atc_codes": [
                {"code": "A01AA01", "level": 5, "classification": {}, "indication": "", "confidence": 0.0, "reasoning": "", "verified": true}
                ],
                "has_multiple_codes": false
            },
            "evidence_summary": [{"source": "who_atc", "claim": ""}],
            "needs_human_review": false,
            "review_reason": ""
            }
            """.strip()

        kwargs = {
            "model": model,
            "tools": make_atc_tools(),
            "system_prompt": system_prompt,
            "backend": backend,
            # 技能根目录即 backend 根目录；传入根路径可发现其下多个 Skill 子目录。
            "skills": ["/"],
            # 只暴露 Skill 所需的只读文件工具，不提供 shell、写入或删除。
            "middleware": [
                FilesystemMiddleware(
                    backend=backend, tools=["read_file", "ls", "glob", "grep"]
                )
            ],
        }
        # 默认不启用通用子 Agent（避免每个药物都产生额外模型调用）。
        # deepagents 0.7.x 起 create_deep_agent 改用 subagents 参数。
        if not AGENT_CONFIG["enable_subagents"]:
            kwargs["subagents"] = []
        return create_deep_agent(**kwargs)

    def run(self, drug_name: str) -> Dict:
        """执行一轮映射；证据不足时在预算内再次调用 Agent 精炼结论。"""
        rounds: List[Dict] = []
        prompt = f"请为药物“{drug_name}”进行 WHO ATC 映射。"
        result, execution = self._invoke(prompt)
        rounds.append(execution)

        refinement_count = 0
        while (
            self._needs_refinement(result)
            and refinement_count < self.max_refinement_rounds
        ):
            refinement_count += 1
            prompt = self._refinement_prompt(drug_name, result)
            refined_result, execution = self._invoke(prompt)
            execution["refinement_round"] = refinement_count
            rounds.append(execution)
            result = self._prefer_result(result, refined_result)

        atc_result = result.get("atc_result", result)
        atc_codes = atc_result.get("atc_codes", []) if isinstance(atc_result, dict) else []
        return {
            "drug_name": drug_name,
            "normalized": result.get("normalized", {}),
            "atc_result": atc_result if isinstance(atc_result, dict) else {"atc_codes": []},
            "primary_atc_code": atc_codes[0].get("code", "") if atc_codes else "",
            "all_atc_codes": [item.get("code", "") for item in atc_codes],
            "info_sources": self._sources(result),
            "needs_human_review": result.get("needs_human_review", not bool(atc_codes)),
            "review_reason": result.get("review_reason", ""),
            "status": "success",
            "deep_agent_trace": {
                "runtime": "deepagents",
                "rounds": rounds,
                "refinement_count": refinement_count,
            },
        }

    def _invoke(self, prompt: str) -> Tuple[Dict, Dict]:
        response = self.agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        final_text = self._final_text(response)
        parsed = self._parse_json(final_text)
        trace = {
            "tool_calls": self._tool_calls(response),
            "final_text": final_text[:4000],
            "parse_ok": bool(parsed),
        }
        if not parsed:
            logger.warning("Deep Agents 输出不是预期 JSON，将标记为待人工核验")
            parsed = {
                "normalized": {},
                "atc_result": {"atc_codes": []},
                "needs_human_review": True,
                "review_reason": "模型未返回有效 JSON",
            }
        return parsed, trace

    @staticmethod
    def _final_text(response: Dict) -> str:
        messages = response.get("messages", []) if isinstance(response, dict) else []
        for message in reversed(messages):
            content = getattr(message, "content", "")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
                ]
                if text_parts:
                    return "\n".join(text_parts)
        return ""

    @staticmethod
    def _parse_json(content: str) -> Dict:
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

    @staticmethod
    def _tool_calls(response: Dict) -> List[Dict]:
        calls = []
        for message in response.get("messages", []) if isinstance(response, dict) else []:
            for call in getattr(message, "tool_calls", []) or []:
                calls.append(
                    {"name": call.get("name", ""), "args": call.get("args", {})}
                )
        return calls

    def _needs_refinement(self, result: Dict) -> bool:
        codes = result.get("atc_result", result).get("atc_codes", [])
        if not codes:
            return True
        confidences = []
        for item in codes:
            code = item.get("code", "")
            if not ATC_CODE_PATTERN.match(code):
                return True
            confidences.append(float(item.get("confidence", 0) or 0))
        return max(confidences, default=0) < self.refinement_threshold

    @staticmethod
    def _prefer_result(current: Dict, candidate: Dict) -> Dict:
        """只接受候选编码更多或最高置信度更高的精炼结果，避免无效回退。"""
        def score(value: Dict) -> Tuple[int, float]:
            codes = value.get("atc_result", value).get("atc_codes", [])
            valid = [
                code for code in codes
                if ATC_CODE_PATTERN.match(code.get("code", ""))
            ]
            confidence = max((float(c.get("confidence", 0) or 0) for c in valid), default=0)
            return len(valid), confidence

        return candidate if score(candidate) > score(current) else current

    @staticmethod
    def _sources(result: Dict) -> List[str]:
        evidence = result.get("evidence_summary", [])
        sources = [item.get("source") for item in evidence if isinstance(item, dict)]
        return sorted({source for source in sources if source})

    @staticmethod
    def _refinement_prompt(drug_name: str, result: Dict) -> str:
        return (
            f"请复核药物“{drug_name}”的上一轮结果。当前结果如下：\n"
            f"{json.dumps(result, ensure_ascii=False)}\n\n"
            "找出证据缺口，只调用必要的补充工具；尤其要核验候选 ATC。"
            "若仍缺少权威证据，保留空候选或降低置信度并标记 needs_human_review。"
            "最终仍只输出规定 JSON。"
        )
