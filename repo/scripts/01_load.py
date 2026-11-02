"""
01_load.py —— 取数 / 读入并检视
================================
作用：
  1. 读取原始 Superstore 数据（支持 .xls 与 .xlsx，自动识别）
  2. 打印每张表的 shape / 列名 / 数据类型 / 缺失值 —— 这些是论文「数据集概述」「数据清洗」要抄的数字
  3. 把三张表导出为 data/clean/*.csv 和一份 superstore.sqlite，满足「数据可导出复用」

运行：
  cd repo
  pip install -r requirements.txt
  python scripts/01_load.py

注意：
  - .xls 需要 xlrd>=2.0.1（pandas 3.x 兼容版本），已在 requirements.txt 写明
  - 本脚本只「读和存」，不做任何清洗；清洗在 02_clean.py
"""

from pathlib import Path
import sqlite3
import sys

import pandas as pd

# ---- 路径配置（兼容两种运行方式）----
# 作为脚本运行：python scripts/01_load.py   -> 自动定位 repo/
# 在 Jupyter cell 里直接粘贴运行              -> 用下面的硬编码路径
try:
    REPO = Path(__file__).resolve().parents[1]      # repo/
except NameError:  # Jupyter / 交互式环境没有 __file__
    REPO = Path(r"C:/Users/LENOVO/Desktop/repo")     # ← 改成你的 repo 实际位置
RAW_DIR = REPO / "data" / "raw"
CLEAN_DIR = REPO / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# 自动找到原始文件（.xls 或 .xlsx 都行）
xls_files = sorted(list(RAW_DIR.glob("*.xls")) + list(RAW_DIR.glob("*.xlsx")))
if not xls_files:
    sys.exit(f"[错误] 在 {RAW_DIR} 下没找到任何 .xls/.xlsx 文件，请先把数据放进去。")
RAW_PATH = xls_files[0]
print(f"[info] 找到原始文件：{RAW_PATH.name}")

# ---- 读取所有工作表 ----
# .xls 由 pandas 自动调用 xlrd；.xlsx 调用 openpyxl
xls = pd.ExcelFile(RAW_PATH)
sheet_names = xls.sheet_names
print(f"[info] 工作表名：{sheet_names}")

tables = {}
for sh in sheet_names:
    df = xls.parse(sh)
    tables[sh] = df

    print("\n" + "=" * 60)
    print(f"表名：{sh}    形状：{df.shape[0]} 行 × {df.shape[1]} 列")
    print("-" * 60)
    print("列名：")
    for c in df.columns:
        print(f"   - {c}")
    print("数据类型：")
    print(df.dtypes.to_string())
    print("缺失值（按列，>0 才显示）：")
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if len(miss) == 0:
        print("   无缺失值")
    else:
        print(miss.to_string())

# ---- 导出 CSV ----
print("\n" + "=" * 60)
print("[导出] 写入 data/clean/*.csv ...")
for sh, df in tables.items():
    out = CLEAN_DIR / f"{sh}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"   -> {out.name}  ({df.shape[0]} 行)")

# ---- 导出 SQLite ----
db_path = CLEAN_DIR / "superstore.sqlite"
conn = sqlite3.connect(db_path)
for sh, df in tables.items():
    df.to_sql(sh, conn, if_exists="replace", index=False)
conn.close()
print(f"[导出] 写入 {db_path.name}")

print("\n[完成] 取数阶段结束。请把上面每张表的 shape / 列名 / 缺失值 抄进论文。")
print("[下一步] 运行 02_clean.py 做清洗。")
