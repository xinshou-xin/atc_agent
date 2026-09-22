"""
向量库构建脚本（本地 CSV 知识库）
用法:
    python build_vector_db.py                     # 灌入 data/knowledge/ATC.csv（默认）
    python build_vector_db.py path/to/other.csv   # 灌入指定 CSV
    python build_vector_db.py --preview           # 只预览列映射和样例，不写库

注意:
    - chroma 重复 id 会自动跳过，重复运行不会灌重
    - 想重建库：先删掉 data/vector_db/ 目录再跑
"""
import os
import sys
import hashlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import PROJECT_ROOT, VECTOR_DB
from retriever.vector_db import VectorDB
from utils.logger import setup_logger

logger = setup_logger("build_vector_db")

DEFAULT_CSV = os.path.join(PROJECT_ROOT, "data", "knowledge", "ATC.csv")
BATCH_SIZE = 100  # 分批写入，避免一次性向量化过多文本


# ============================================================
# 模式一：本地 CSV 知识文件（自动探测编码和列名）
# ============================================================

# 列名候选（按优先级），支持中英文常见命名
COLUMN_CANDIDATES = {
    "drug_name": ["drug_name", "drugname", "drug", "name", "drug_name_en",
                  "药物名", "药物名称", "药品名", "名称", "通用名"],
    "atc_code": ["atc_code", "atccode", "atc", "code", "atc_code_5",
                 "atc编码", "编码", "atc码"],
    "description": ["description", "desc", "info", "indication", "pharmacology",
                    "summary", "描述", "说明", "适应症", "药理", "备注"],
}


def load_csv(path: str):
    """读取 CSV，自动尝试常见编码"""
    import pandas as pd

    for enc in ["utf-8-sig", "utf-8", "gbk", "latin-1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info(f"CSV 读取成功（编码: {enc}），共 {len(df)} 行")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"CSV 读取失败: {e}")
            raise

    raise ValueError(f"无法识别文件编码: {path}")


def detect_columns(df) -> dict:
    """根据列名模糊匹配，返回 {逻辑列名: 实际列名}"""
    actual_cols = {str(c).strip().lower(): c for c in df.columns}
    mapping = {}
    for logical, candidates in COLUMN_CANDIDATES.items():
        for cand in candidates:
            if cand.lower() in actual_cols:
                mapping[logical] = actual_cols[cand.lower()]
                break
    return mapping


def clean(value) -> str:
    """单元格转干净字符串，NaN/空 → ''"""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none", "") else s


def build_from_csv(path: str, preview_only: bool = False):
    df = load_csv(path)
    mapping = detect_columns(df)

    print("\n" + "=" * 60)
    print("列名映射结果（逻辑列 → 实际列）:")
    for logical in COLUMN_CANDIDATES:
        actual = mapping.get(logical, "【未找到】")
        print(f"  {logical:<12} -> {actual}")
    print("=" * 60)

    # 至少要有一列（药物名或编码），否则没法组装知识
    if "drug_name" not in mapping and "atc_code" not in mapping:
        logger.error("未识别出药物名/ATC 编码列，请检查 CSV 表头")
        print(f"实际表头: {list(df.columns)}")
        return [], [], []

    col_name = mapping.get("drug_name")
    col_code = mapping.get("atc_code")
    col_desc = mapping.get("description")

    texts, ids, metas = [], [], []
    skipped = 0

    for i, row in df.iterrows():
        name = clean(row[col_name]) if col_name else ""
        atc = clean(row[col_code]) if col_code else ""
        desc = clean(row[col_desc]) if col_desc else ""

        if not name and not atc:
            skipped += 1
            continue

        # 组装知识文档
        doc = f"药物: {name}" if name else "药物: (未知)"
        if atc:
            doc += f"，ATC 编码: {atc}"
        if desc:
            doc += f"，相关信息: {desc}"

        texts.append(doc)
        # id 用内容哈希，重复运行不产生重复数据
        key = f"{name}|{atc}".encode("utf-8")
        ids.append("kb_" + hashlib.md5(key).hexdigest()[:16])
        metas.append({
            "source": "atc_csv",
            "drug": name[:100],
            "code": atc[:20],
        })

    if skipped:
        logger.warning(f"跳过 {skipped} 行空记录")

    # 预览模式：打印前几条样例
    if preview_only:
        print(f"\n共组装 {len(texts)} 条知识文档，前 3 条样例:")
        for t in texts[:3]:
            print(f"  - {t}")
        return [], [], []

    return texts, ids, metas


# ============================================================
# 入口
# ============================================================
def add_in_batches(db: VectorDB, texts, ids, metas):
    """分批写入向量库"""
    total = len(texts)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        db.add(texts[start:end], ids[start:end], metas[start:end])
        logger.info(f"进度: {end}/{total}")


def _wait_chroma_persist(db: VectorDB) -> None:
    """灌库后等待 Chroma compactor 完成 HNSW 索引落盘。

    注意：不能在灌库进程内重新打开同一路径的 PersistentClient（会干扰
    compactor 写盘，导致索引未落盘）。本函数只做固定等待，让后台 compactor
    完成分片索引写盘后进程再退出。
    """
    if VECTOR_DB.get("provider") != "chroma":
        return

    import time

    logger.info("等待 Chroma 索引落盘（300 秒）...")
    time.sleep(300)
    logger.info("Chroma 索引落盘等待结束")


if __name__ == "__main__":
    args = sys.argv[1:]
    mode = args[0] if args else "file"
    preview = "--preview" in args
    db = None

    if mode == "file":
        # 第二个非 --preview 参数作为 CSV 路径
        csv_path = next((a for a in args[1:] if a != "--preview"), DEFAULT_CSV)
        logger.info(f"知识文件: {csv_path}")
        texts, ids, metas = build_from_csv(csv_path, preview_only=preview)
        if not preview and texts:
            db = VectorDB()
            add_in_batches(db, texts, ids, metas)

    else:
        print("用法: python build_vector_db.py [file|who] [csv路径] [--preview]")
        sys.exit(1)

    if db:
        logger.info(f"灌库完成，库内共 {db.count()} 条")
        _wait_chroma_persist(db)
