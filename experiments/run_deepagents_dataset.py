# -*- coding: utf-8 -*-
"""
run_deepagents_dataset.py
=========================
用五大数据集对 DeepATCAgent（deepagents 运行时）做批量评测。

与 run_pyxis_experiment.py 的区别：
    被测对象是 DeepATCAgent（动态规划 + Skills + 工具调用循环），
    而不是 legacy DrugAgent + ReflectionAgent 流程。

输出：
    results/deepagents/{dataset}/predictions.jsonl   每条药物一行（金标准/预测/来源/耗时）
    results/deepagents/_report.xlsx                  汇总指标表（--report 也会刷新）

用法：
    cd experiments
    python run_deepagents_dataset.py --dataset gdph --limit 3        # 中文集冒烟测试
    python run_deepagents_dataset.py --dataset pyxis                 # 单个数据集全量
    python run_deepagents_dataset.py --dataset all --limit 20        # 每个数据集各 20 个
    python run_deepagents_dataset.py --dataset all                   # 全部 5 个数据集
    python run_deepagents_dataset.py --dataset gdph --resume         # 断点续跑
    python run_deepagents_dataset.py --report                        # 只汇总已有结果
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                   # experiments/
sys.path.insert(0, os.path.dirname(_HERE))                  # drug_agent/

import pandas as pd                                         # noqa: E402

from dataset_loader import DATASETS, load_dataset           # noqa: E402
from metrics import (                                       # noqa: E402
    evaluate, evaluate_by_cardinality, load_kb_codes, classify_match, clean_codes,
)

RESULT_DIR = os.path.join(_HERE, "results", "deepagents")
REPORT_XLSX = os.path.join(RESULT_DIR, "_report.xlsx")


# ============================================================
# 缓存隔离：必须在实例化 DeepATCAgent 之前调用
# ============================================================
def isolate_cache(dataset: str) -> str:
    """把全局缓存目录切到 results/deepagents/{dataset}/cache，各数据集互不污染。"""
    from config import settings
    cache_dir = os.path.join(RESULT_DIR, dataset, "cache")
    settings.CACHE["dir"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


# ============================================================
# JSONL 工具
# ============================================================
def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_done(path: str) -> dict:
    """读取已完成且无错误的预测，返回 {药物名: 记录}"""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("error"):            # 失败的重跑
                continue
            done[obj["name"]] = obj
    return done


def fmt_eta(elapsed: float, done: int, total: int) -> str:
    if done == 0:
        return "--"
    remain = elapsed / done * (total - done)
    return f"{int(remain // 60)}分{int(remain % 60)}秒"


# ============================================================
# 批量运行 DeepATCAgent
# ============================================================
def run_deepagent(records: list, out_path: str, resume: bool):
    from agents.deep_atc_agent import DeepATCAgent

    agent = DeepATCAgent()
    done = load_done(out_path) if resume else {}
    if done:
        print(f"[续跑] 已完成 {len(done)} 条，跳过")

    total = len(records)
    t_start = time.time()

    for i, r in enumerate(records, 1):
        name = r["name"]
        if name in done:
            continue

        t0 = time.time()
        rec = {
            "name": name,
            "dataset": r["dataset"],
            "lang": r["lang"],
            "gold_codes": r["gold_codes"],
        }
        try:
            result = agent.run(name)

            # 预测编码（与 all_atc_codes 同序），置信度从 atc_result 对齐
            items = (result.get("atc_result") or {}).get("atc_codes") or []
            conf_map = {}
            for it in items:
                c = str(it.get("code", "") or "").strip().upper()
                if c:
                    try:
                        conf_map[c] = float(it.get("confidence", 0.5) or 0.5)
                    except (TypeError, ValueError):
                        conf_map[c] = 0.5
            pred_codes = [str(c).strip().upper() for c in (result.get("all_atc_codes") or [])
                          if str(c).strip()]

            rec.update({
                "pred_codes": pred_codes,
                "pred_confs": [conf_map.get(c, 0.5) for c in pred_codes],
                "primary_atc_code": str(result.get("primary_atc_code") or "").upper(),
                "info_sources": result.get("info_sources") or [],
                "needs_human_review": bool(result.get("needs_human_review")),
                "review_reason": result.get("review_reason") or "",
                "error": "",
            })
        except Exception as e:
            rec.update({
                "pred_codes": [], "pred_confs": [], "primary_atc_code": "",
                "info_sources": [], "needs_human_review": True,
                "review_reason": "", "error": f"{type(e).__name__}: {e}",
            })

        rec["elapsed_s"] = round(time.time() - t0, 2)
        append_jsonl(out_path, rec)

        if i % 5 == 0 or i == total:
            el = time.time() - t_start
            print(f"  [{i}/{total}] ETA {fmt_eta(el, i, total)} | "
                  f"{name[:26]} -> {rec['pred_codes']} | src={rec['info_sources']}")

    return load_done(out_path)


# ============================================================
# 汇总报告
# ============================================================
def report(ds_list: list):
    kb = load_kb_codes()
    all_rows, details = [], {}

    for ds in ds_list:
        path = os.path.join(RESULT_DIR, ds, "predictions.jsonl")
        if not os.path.exists(path):
            print(f"[跳过] {ds}: 无结果文件")
            continue
        with open(path, "r", encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
        if not recs:
            print(f"[跳过] {ds}: 结果为空")
            continue

        n_err = sum(1 for r in recs if r.get("error"))
        n_review = sum(1 for r in recs if r.get("needs_human_review"))
        res = evaluate(recs, kb_codes=kb)

        src_counter = {}
        for r in recs:
            for s in r.get("info_sources") or []:
                src_counter[s] = src_counter.get(s, 0) + 1

        res.update({
            "dataset": ds,
            "lang": recs[0].get("lang", ""),
            "n_done": len(recs),
            "n_error": n_err,
            "review_rate": round(n_review / len(recs), 4),
            "avg_elapsed": round(sum(r.get("elapsed_s", 0) for r in recs) / len(recs), 2),
            "source_hits": src_counter,
        })
        all_rows.append(res)
        details[ds] = recs

    if not all_rows:
        print("\n还没有任何结果。先跑：python run_deepagents_dataset.py --dataset gdph --limit 3")
        return

    # ---------- 主表 ----------
    print("\n" + "=" * 88)
    print("DeepATCAgent 数据集评测（三分类 + F1 + 人工核验率）")
    print("=" * 88)
    hdr = (f"{'Dataset':<14}{'Lang':>4}{'N':>6}{'Exact':>8}{'Partial':>9}"
           f"{'None':>8}{'F1':>7}{'Top1':>7}{'Halluc':>8}{'Review':>8}")
    print(hdr)
    print("-" * 88)
    for r in all_rows:
        print(f"{r['dataset']:<14}{r['lang']:>4}{r['n_done']:>6}"
              f"{r['exact_match']:>8.3f}{r['partial_match']:>9.3f}"
              f"{r['no_match']:>8.3f}{r['f1']:>7.3f}{r['top1_acc']:>7.3f}"
              f"{r.get('hallucination_rate', 0):>8.3f}{r['review_rate']:>8.3f}")

    # ---------- 体检项 ----------
    print("\n" + "=" * 88)
    print("体检项（排查问题用）")
    print("=" * 88)
    for r in all_rows:
        print(f"\n[{r['dataset']}]")
        print(f"  调用失败: {r['n_error']}   (应接近 0)")
        print(f"  需人工核验: {r['review_rate']:.1%}   (过高说明置信度阈值/规则要调)")
        print(f"  平均耗时: {r['avg_elapsed']}s / 药")
        print(f"  检索源命中: {r['source_hits'] or '(无)'}")
        print(f"  覆盖率: {r['coverage']:.3f}")

    # ---------- 一对多分层（仅对第一个有结果的数据集，避免刷屏） ----------
    first = next((r["dataset"] for r in all_rows), None)
    if first:
        print("\n" + "=" * 88)
        print(f"按金标准编码数分层（一对多难度，{first}）")
        print("=" * 88)
        print(evaluate_by_cardinality(details[first], kb).to_string(index=False))

    # ---------- 落盘 ----------
    # 固定名 _report.xlsx = 最新全量汇总（方便直接查看）；
    # 时间戳快照 _report_YYYYmmdd_HHMMSS.xlsx = 保留每次运行的历史，避免被覆盖。
    os.makedirs(os.path.dirname(REPORT_XLSX), exist_ok=True)
    snapshot = os.path.join(
        os.path.dirname(REPORT_XLSX),
        f"_report_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
    )

    def _write_report(path: str):
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            pd.DataFrame([{k: v for k, v in r.items() if k != "source_hits"}
                          for r in all_rows]).to_excel(w, sheet_name="main", index=False)
            for ds, recs in details.items():
                pd.DataFrame([{
                    "name": r["name"],
                    "gold": "|".join(r["gold_codes"]),
                    "pred": "|".join(r.get("pred_codes", [])),
                    "primary": r.get("primary_atc_code", ""),
                    "match_class": classify_match(
                        set(clean_codes(r["gold_codes"])),
                        set(clean_codes(r.get("pred_codes", [])))),
                    "needs_human_review": r.get("needs_human_review", False),
                    "sources": ",".join(r.get("info_sources", []) or []),
                    "error": r.get("error", ""),
                } for r in recs]).to_excel(w, sheet_name=f"detail_{ds}"[:31], index=False)

    _write_report(REPORT_XLSX)
    _write_report(snapshot)
    print(f"\n结果已保存: {REPORT_XLSX}")
    print(f"历史快照: {snapshot}")


# ============================================================
# 主入口
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pyxis",
                    help="pyxis / medrecon / mcomed / gdph / healthcanada / all（默认 pyxis）")
    ap.add_argument("--limit", type=int, default=0, help="每个数据集只跑 N 条（0=全量）")
    ap.add_argument("--offset", type=int, default=0,
                    help="从第 N 条开始（配合 --limit 分段跑；分段时务必加 --resume 保留已有结果）")
    ap.add_argument("--resume", action="store_true", help="断点续跑（跳过已完成且无错误的记录）")
    ap.add_argument("--report", action="store_true", help="不跑实验，只汇总已有结果")
    args = ap.parse_args()

    if args.report:
        ds_list = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
        report(ds_list)
        return

    # 快速失败：避免无 Key 空转
    from config.settings import LLM_CONFIG
    if not LLM_CONFIG.get("api_key"):
        print("\n[错误] 未配置 LLM_API_KEY，请在 drug_agent/.env 中填写")
        sys.exit(1)

    ds_list = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in ds_list:
        if ds not in DATASETS:
            print(f"[错误] 未知 dataset: {ds}，可选 {list(DATASETS)} 或 all")
            sys.exit(1)

    for ds in ds_list:
        records = load_dataset(ds)
        records = records[args.offset:]
        if args.limit > 0:
            records = records[:args.limit]

        print("=" * 88)
        print(f"DeepATCAgent 评测 | 数据集 {ds} | 待测 {len(records)} 个药物")
        print("=" * 88)

        cache_dir = isolate_cache(ds)               # 必须在实例化 Agent 前
        print(f">>> 缓存目录: {cache_dir}")

        out_dir = os.path.join(RESULT_DIR, ds)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "predictions.jsonl")

        if not args.resume and os.path.exists(out_path):
            bak = out_path + f".bak_{int(time.time())}"
            os.rename(out_path, bak)
            print(f"    旧结果已归档: {os.path.basename(bak)}")

        t0 = time.time()
        run_deepagent(records, out_path, args.resume)
        print(f"<<< {ds} 完成，耗时 {int(time.time() - t0)} 秒")

    report(ds_list)


if __name__ == "__main__":
    main()
