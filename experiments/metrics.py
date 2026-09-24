# -*- coding: utf-8 -*-
"""
metrics.py
==========
ATC 映射评测指标（集合级，适配一对多金标准）。

核心三分类（互斥且穷尽，占比之和 = 100%）：
    exact    精准匹配 —— 预测集合 == 金标准集合
    partial  部分匹配 —— 有交集但不相等（漏码或多码）
    none     不匹配   —— 交集为空（含未输出任何编码）

配套指标：Set-Precision / Set-Recall / F1（宏平均）、覆盖率、幻觉率。

用法：
    from metrics import classify_match, score_one, evaluate, load_kb_codes
"""
import os
import re
from typing import Dict, List, Optional, Set

import pandas as pd

# 知识库路径：环境变量可覆盖，否则推断为「drug_agent 上一级 / medication / ATC.xlsx」
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
KB_PATH = os.environ.get("MED_ATC_KB_PATH") or os.path.join(_PROJECT, "medication", "ATC.xlsx")


# ---------------------------------------------------------------- 知识库
def load_kb_codes(path: Optional[str] = None) -> Set[str]:
    """加载 WHO ATC 知识库编码集合（用于幻觉检测）。文件不存在则返回空集。"""
    p = path or KB_PATH
    if not os.path.exists(p):
        return set()
    df = pd.read_excel(p)
    return set(df["code"].astype(str).str.strip().str.upper())


def _norm(code) -> str:
    if code is None:
        return ""
    s = str(code).strip().upper()
    return "" if s in ("NAN", "NONE", "NA") else s


def clean_codes(codes) -> List[str]:
    """清洗编码列表：去空、大写、去重、保序。"""
    out, seen = [], set()
    for c in codes or []:
        s = _norm(c)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ---------------------------------------------------------------- 三分类
def classify_match(gold: Set[str], pred: Set[str]) -> str:
    """
    返回 "exact" / "partial" / "none"

    gold 恒非空（无 ATC 的行已在数据加载阶段过滤）。
    pred 为空集 → "none"（未给出任何编码）。
    """
    if not pred:
        return "none"
    if pred == gold:
        return "exact"
    if pred & gold:
        return "partial"
    return "none"


# ---------------------------------------------------------------- 单药物打分
def score_one(gold_codes, pred_codes, kb_codes: Optional[Set[str]] = None) -> Dict:
    """计算单个药物的全部指标。"""
    gold = set(clean_codes(gold_codes))
    pred = clean_codes(pred_codes)          # 保留顺序（置信度排序）
    pred_set = set(pred)

    tp = len(pred_set & gold)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    halluc = [c for c in pred if kb_codes is not None and c not in kb_codes]

    return {
        "n_gold": len(gold),
        "n_pred": len(pred),
        "tp": tp,
        "match_class": classify_match(gold, pred_set),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "top1_hit": float(bool(pred) and pred[0] in gold),
        "prefix_hit": float(any(
            p.startswith(g) or g.startswith(p) for p in pred for g in gold
        )),
        "n_halluc": len(halluc),
        "halluc_codes": halluc,
    }


# ---------------------------------------------------------------- 汇总
def evaluate(items: List[Dict], kb_codes: Optional[Set[str]] = None) -> Dict:
    """
    宏平均汇总。items 每项需含 gold_codes / pred_codes。
    返回可直接填论文表格的指标字典。
    """
    n = len(items)
    if n == 0:
        return {"n": 0}

    scored = [score_one(it["gold_codes"], it["pred_codes"], kb_codes) for it in items]

    cls_count = {"exact": 0, "partial": 0, "none": 0}
    for s in scored:
        cls_count[s["match_class"]] += 1

    total_pred = sum(s["n_pred"] for s in scored)
    total_halluc = sum(s["n_halluc"] for s in scored)
    mean = lambda k: sum(s[k] for s in scored) / n

    res = {
        "n_drugs": n,
        # —— 用户要的四个核心指标 ——
        "exact_match": round(cls_count["exact"] / n, 4),
        "partial_match": round(cls_count["partial"] / n, 4),
        "no_match": round(cls_count["none"] / n, 4),
        "f1": round(mean("f1"), 4),
        # —— 配套指标 ——
        "precision": round(mean("precision"), 4),
        "recall": round(mean("recall"), 4),
        "top1_acc": round(mean("top1_hit"), 4),
        "prefix_acc": round(mean("prefix_hit"), 4),
        "coverage": round(sum(1 for s in scored if s["n_pred"] > 0) / n, 4),
        "n_exact": cls_count["exact"],
        "n_partial": cls_count["partial"],
        "n_none": cls_count["none"],
    }
    if kb_codes:
        res["hallucination_rate"] = round(total_halluc / total_pred, 4) if total_pred else 0.0
        res["n_halluc_codes"] = total_halluc
    return res


def evaluate_by_cardinality(items: List[Dict], kb_codes: Optional[Set[str]] = None) -> pd.DataFrame:
    """按金标准编码数分层统计（一对多难度分析）。"""
    buckets = {}
    for it in items:
        k = len(set(clean_codes(it["gold_codes"])))
        key = "1" if k == 1 else ("2" if k == 2 else ("3" if k == 3 else "4+"))
        buckets.setdefault(key, []).append(it)

    rows = []
    for key in ["1", "2", "3", "4+"]:
        sub = buckets.get(key, [])
        if not sub:
            continue
        r = evaluate(sub, kb_codes)
        rows.append({
            "Gold codes": key,
            "Drugs": r["n_drugs"],
            "Exact": r["exact_match"],
            "Partial": r["partial_match"],
            "None": r["no_match"],
            "F1": r["f1"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 自检
if __name__ == "__main__":
    print("=" * 60)
    print("metrics.py 自检")
    print("=" * 60)

    kb = load_kb_codes()
    print(f"知识库编码数: {len(kb)}")

    cases = [
        (["N02BA01"], ["N02BA01"], "exact"),
        (["B01AC06", "N02BA01"], ["B01AC06"], "partial"),      # 漏一个
        (["B01AC06"], ["B01AC06", "N02BA01"], "partial"),       # 多一个
        (["N02BA01"], ["B01AC06"], "none"),                     # 全错
        (["N02BA01"], [], "none"),                              # 没输出
    ]
    ok = True
    for gold, pred, expect in cases:
        got = classify_match(set(clean_codes(gold)), set(clean_codes(pred)))
        flag = "OK" if got == expect else "FAIL"
        if got != expect:
            ok = False
        print(f"  [{flag}] gold={gold} pred={pred} -> {got} (期望 {expect})")

    # 幻觉检测
    s = score_one(["N02BA01"], ["N02BA01", "Z99ZZ99"], kb)
    print(f"\n幻觉检测: pred=['N02BA01','Z99ZZ99'] -> n_halluc={s['n_halluc']} "
          f"codes={s['halluc_codes']} (期望 1 个 Z99ZZ99)")

    # 汇总
    items = [{"gold_codes": g, "pred_codes": p} for g, p, _ in cases]
    print("\n汇总（5 条构造样本）:")
    for k, v in evaluate(items, kb).items():
        print(f"  {k:<20} {v}")

    print("\n自检结果:", "全部通过" if ok else "存在失败")
