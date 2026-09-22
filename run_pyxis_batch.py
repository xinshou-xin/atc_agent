# -*- coding: utf-8 -*-
"""批量跑药物表 ATC 映射（Deep Agent），结果保存为 Excel + JSONL。

用法示例:
    python run_pyxis_batch.py --input data/pyxis.csv --limit 20          # 跑前 20 个
    python run_pyxis_batch.py --input data/pyxis.xlsx --limit 50 --offset 20   # 从第 20 个继续
    python run_pyxis_batch.py --input data/pyxis.csv --no-dedup          # 不去重（按原表行）
    python run_pyxis_batch.py --input data/pyxis.csv --limit 100 --resume  # 跳过已跑过的药物名

说明:
    - 默认按药物名去重后再跑（同一药物只映射一次，避免浪费 LLM 调用）
    - 每条结果即时写入 JSONL（data/output/pyxis_atc_<时间戳>.jsonl），中途中断不丢
    - 全部跑完后生成汇总 Excel（data/output/pyxis_atc_<时间戳>.xlsx）
    - --resume 会扫描 output 目录里已有的 JSONL，跳过其中已完成的药物名
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.deep_atc_agent import DeepATCAgent


def load_names(path: str, dedup: bool):
    """读取药物名列表（自动识别 csv/xlsx），返回 (names, 原始列名)。"""
    import pandas as pd
    if str(path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    col = "name" if "name" in df.columns else df.columns[0]
    names = df[col].dropna().astype(str).str.strip()
    names = names[names != ""]
    if dedup:
        names = names.drop_duplicates()
    return list(names), list(df.columns)


def extract_result(r: dict) -> dict:
    """从 agent.run() 结果中提取结构化字段。"""
    return {
        "drug_name": r.get("drug_name", ""),
        "primary_atc_code": r.get("primary_atc_code", ""),
        "atc_all": ";".join(r.get("all_atc_codes") or []),
        "atc_count": len(r.get("all_atc_codes") or []),
        "info_sources": ";".join(r.get("info_sources") or []),
        "needs_human_review": r.get("needs_human_review"),
        "review_reason": r.get("review_reason", ""),
        "status": r.get("status", ""),
        "error": "",
    }


def load_done_names(out_dir: Path) -> set:
    """扫描 output 目录已有 JSONL，返回已完成的药物名集合（用于断点续跑）。"""
    done = set()
    for jl in sorted(out_dir.glob("pyxis_atc_*.jsonl")):
        try:
            with open(jl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done.add(json.loads(line).get("drug_name", ""))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return {d for d in done if d}


def main():
    ap = argparse.ArgumentParser(description="批量跑药物 ATC 映射")
    ap.add_argument("--input", default="data/pyxis.csv", help="输入表路径（csv/xlsx）")
    ap.add_argument("--output-dir", default="data/output", help="输出目录")
    ap.add_argument("--limit", type=int, default=0, help="最多跑 N 条；0=全部")
    ap.add_argument("--offset", type=int, default=0, help="跳过前 N 条")
    ap.add_argument("--no-dedup", action="store_true", help="不去重（按原表行跑）")
    ap.add_argument("--resume", action="store_true", help="跳过已跑过的药物名")
    args = ap.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"[错误] 输入表不存在: {input_path}")
        sys.exit(1)

    names, cols = load_names(input_path, dedup=not args.no_dedup)
    total = len(names)
    print(f"输入表: {input_path}（共 {total} 个药物名，去重={'是' if not args.no_dedup else '否'}）")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = out_dir / f"pyxis_atc_{stamp}.jsonl"
    xlsx_path = out_dir / f"pyxis_atc_{stamp}.xlsx"

    done = load_done_names(out_dir) if args.resume else set()
    if done:
        print(f"断点续跑：已跳过 {len(done)} 个之前完成的药物名")

    print("加载 Deep Agent（首次会加载 bge-m3 模型，约 10~20 秒）...")
    agent = DeepATCAgent()

    rows = []
    ran = 0
    start = time.time()
    for i, name in enumerate(names):
        if i < args.offset:
            continue
        if args.limit and ran >= args.limit:
            print(f"达到 --limit {args.limit}，停止。")
            break
        if name in done:
            continue

        t0 = time.time()
        try:
            r = agent.run(name)
            row = extract_result(r)
        except Exception as e:
            row = {
                "drug_name": name,
                "primary_atc_code": "",
                "atc_all": "",
                "atc_count": 0,
                "info_sources": "",
                "needs_human_review": True,
                "review_reason": "",
                "status": "error",
                "error": str(e)[:500],
            }
        rows.append(row)
        ran += 1

        # 即时落盘 JSONL（断点续跑依赖）
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        elapsed = time.time() - start
        per = elapsed / ran if ran else 0
        eta = per * (total - args.offset - ran) if ran else 0
        mark = "✓" if row.get("status") != "error" else "✗"
        print(
            f"[{i + 1 - args.offset}/{total - args.offset}] {name[:40]:<42} "
            f"{mark} atc={row.get('atc_count', 0)} 用时{time.time() - t0:.1f}s "
            f"累计{elapsed / 60:.1f}min 剩余约{eta / 60:.1f}min"
        )
        sys.stdout.flush()

    # 汇总 Excel
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_excel(xlsx_path, index=False)
        ok = int((df["status"] != "error").sum()) if "status" in df else len(df)
        print(f"\n完成: 成功 {ok}/{len(df)}，失败 {len(df) - ok}")
        print(f"结果 Excel : {xlsx_path}")
        print(f"明细 JSONL : {jsonl_path}")
    else:
        print("本次没有新跑任何药物（可能已全部完成）。")


if __name__ == "__main__":
    main()
