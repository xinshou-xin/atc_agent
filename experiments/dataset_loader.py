# -*- coding: utf-8 -*-
"""
dataset_loader.py
=================
将 Data_version2 下五个数据集统一加载为标准评测记录。

每条记录格式：
    {
        "name": str,            # 原始药物名称（脏输入）
        "gold_codes": List[str],# 金标准 ATC 编码集合（去重、大写、去空）
        "dataset": str,         # 数据集名
        "lang": str,            # "en" / "zh"
        "occurrence": int,      # 出现次数
        "patient": int,         # 患者数
    }

清洗规则（V3 规划口径）：
    1. 无 ATC 编码的行 → 过滤（不进评测集）
    2. 编码数 > MAX_CODES(默认10) 的行 → 剔除（极端一对多，单独分析）
    3. GDPH 中的 4 级编码保留（评测用前缀匹配）

用法：
    from dataset_loader import load_dataset, load_all_datasets
    records = load_dataset("pyxis")
    all_records = load_all_datasets()
"""
import re
import os
import pandas as pd

# ============ 数据集注册表 ============
# 数据根目录解析顺序：
#   1) 环境变量 MED_ATC_DATA_ROOT（数据放在别处时用这个）
#   2) 项目内 data/dataset（本地布局：drug_deepagengts/data/dataset）
#   3) 服务器布局：<项目上级>/deepagengts_drug/data/dataset
_HERE = os.path.dirname(os.path.abspath(__file__))            # experiments/
_PROJECT = os.path.dirname(_HERE)                             # 项目根（如 drug_deepagengts）
_DATA_CANDIDATES = [
    os.path.join(_PROJECT, "data", "dataset"),
    os.path.join(os.path.dirname(_PROJECT), "deepagengts_drug", "data", "dataset"),
]


def _resolve_data_root() -> str:
    env = os.environ.get("MED_ATC_DATA_ROOT")
    if env:
        return env
    for p in _DATA_CANDIDATES:
        if os.path.isdir(p):
            return p
    return _DATA_CANDIDATES[0]


DATA_ROOT = _resolve_data_root()

DATASETS = {
    "pyxis": {
        "file": os.path.join(DATA_ROOT, "MIMIC-pyxis_ATC.xlsx"),
        "name_col": "name",
        "lang": "en",
    },
    "medrecon": {
        "file": os.path.join(DATA_ROOT, "MIMIC-medrecon_ATC.xlsx"),
        "name_col": "name",
        "lang": "en",
    },
    "mcomed": {
        "file": os.path.join(DATA_ROOT, "MC-MED_ATC.xlsx"),
        "name_col": "Name",           # 注意大写 N
        "lang": "en",
    },
    "gdph": {
        "file": os.path.join(DATA_ROOT, "GDPH_ATC.xlsx"),
        "name_col": "药品名称",        # 中文列名
        "code_col": "ATC_Code",       # GDPH 为单列编码窄表（一行一药一码），按名称聚合
        "lang": "zh",
    },
    "healthcanada": {
        "file": os.path.join(DATA_ROOT, "HealthCanada_Product_ATC.xlsx"),
        "name_col": "BRAND_NAME",     # 商品名列
        "lang": "en",
    },
}

MAX_CODES = 10  # 编码数超过此值的药物剔除出主评测集


def _extract_codes(row) -> list:
    """从宽表行中提取所有 atc_N 编码列的值（去重、大写、去空）"""
    codes, seen = [], set()
    for col, val in row.items():
        # 只要编码列（atc_1, atc_2, ...），不要名称列（atc_name_1, ...）
        if not re.fullmatch(r"atc_\d+", str(col), re.IGNORECASE):
            continue
        if pd.isna(val):
            continue
        code = str(val).strip().upper()
        if code and code != "NAN" and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def load_dataset(ds_name: str, max_codes: int = MAX_CODES) -> list:
    """加载单个数据集，返回标准记录列表"""
    if ds_name not in DATASETS:
        raise ValueError(f"未知数据集: {ds_name}，可选: {list(DATASETS)}")

    cfg = DATASETS[ds_name]
    df = pd.read_excel(cfg["file"])

    # 单列编码模式（如 GDPH：一行一药一码，同一药物多行）→ 按药物名聚合编码
    if cfg.get("code_col"):
        grouped = {}
        for _, row in df.iterrows():
            name = str(row.get(cfg["name_col"], "")).strip()
            if not name or name.lower() == "nan":
                continue
            code = str(row.get(cfg["code_col"], "")).strip().upper()
            if not code or code == "NAN":
                continue
            grouped.setdefault(name, set()).add(code)

        records = []
        for name, codes in grouped.items():
            if len(codes) > max_codes:       # 规则2：极端一对多 → 剔除
                continue
            records.append({
                "name": name,
                "gold_codes": sorted(codes),
                "dataset": ds_name,
                "lang": cfg["lang"],
                "occurrence": 0,
                "patient": 0,
            })
        return records

    # 宽表模式（atc_1 / atc_2 ...）：原逻辑
    records = []
    for _, row in df.iterrows():
        name = str(row.get(cfg["name_col"], "")).strip()
        if not name or name.lower() == "nan":
            continue
        codes = _extract_codes(row)
        if not codes:               # 规则1：无 ATC → 过滤
            continue
        if len(codes) > max_codes:  # 规则2：极端一对多 → 剔除
            continue
        records.append({
            "name": name,
            "gold_codes": codes,
            "dataset": ds_name,
            "lang": cfg["lang"],
            "occurrence": int(row.get("occurrence_count", 0) or 0),
            "patient": int(row.get("patient_count", 0) or 0),
        })
    return records


def load_all_datasets(max_codes: int = MAX_CODES) -> dict:
    """加载全部五个数据集，返回 {数据集名: [记录]}"""
    return {ds: load_dataset(ds, max_codes) for ds in DATASETS}


def summary(all_records: dict) -> pd.DataFrame:
    """生成 Table 1 数据集统计（药物数/编码对数/一对多分布/语言）"""
    rows = []
    for ds, recs in all_records.items():
        n = len(recs)
        total_pairs = sum(len(r["gold_codes"]) for r in recs)
        cnt = {k: 0 for k in ["1", "2", "3", "4-5", "6-10"]}
        for r in recs:
            k = len(r["gold_codes"])
            if k == 1: cnt["1"] += 1
            elif k == 2: cnt["2"] += 1
            elif k == 3: cnt["3"] += 1
            elif k <= 5: cnt["4-5"] += 1
            else: cnt["6-10"] += 1
        rows.append({
            "Dataset": ds,
            "Lang": recs[0]["lang"] if recs else "",
            "Drugs": n,
            "Pairs": total_pairs,
            "Avg codes": round(total_pairs / n, 2) if n else 0,
            **{f"#{k}": v for k, v in cnt.items()},
            "Multi(≥2)%": round(sum(v for k, v in cnt.items() if k != "1") / n * 100, 1) if n else 0,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 自检：加载全部数据集并打印 Table 1
    all_recs = load_all_datasets()
    df = summary(all_recs)
    print(df.to_string(index=False))
    print(f"\n合计: {df['Drugs'].sum()} 个药物, {df['Pairs'].sum()} 条映射对")
