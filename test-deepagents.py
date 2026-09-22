# -*- coding: utf-8 -*-
"""DeepATCAgent 精简版调用示例：只打印核心结果"""
import json
from agents.deep_atc_agent import DeepATCAgent

agent = DeepATCAgent()
r = agent.run("阿司匹林")

# 只输出核心字段；完整结果（含 deep_agent_trace 工具调用轨迹）仍在 r 里可访问
print(json.dumps({
    "drug_name": r.get("drug_name"),
    "primary_atc_code": r.get("primary_atc_code"),
    "all_atc_codes": r.get("all_atc_codes"),
    "normalized": r.get("normalized", {}),
    "needs_human_review": r.get("needs_human_review"),
    "review_reason": r.get("review_reason"),
    "info_sources": r.get("info_sources"),
}, ensure_ascii=False, indent=2))
