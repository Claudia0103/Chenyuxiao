"""
02_clean.py —— 清洗与多表对齐
================================
作用（每一步都打印证据，直接抄进论文「数据清洗与治理」）：
  1. 类型校验：日期列/数值列确认（01 已确认基本正确，这里兜底）
  2. 缺失值：统计并报告（本版本实测无缺失）
  3. 异常值：Quantity<=0 / Sales<0 / 非法日期 剔除；Profit<0 属正常业务（退货/高折扣），保留
  4. 重复值：按 Order ID + Product ID 去重，保留首条
  5. 派生字段：Order Year / Order Month / Profit Margin / Is_Returned
  6. 多表对齐：左连 Returns（按 Order ID）、左连 People（按 Region）
  7. 输出：data/clean/orders_clean.csv

运行：Jupyter 直接粘贴整块运行，或 python scripts/02_clean.py
"""

from pathlib import Path
import pandas as pd

# ---- 路径配置（兼容脚本 / Jupyter）----
try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:
    REPO = Path(r"C:/Users/LENOVO/Desktop/repo")
RAW_DIR = REPO / "data" / "raw"
CLEAN_DIR = REPO / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# ---- 读入（和 01 相同的自动找文件逻辑）----
xls_files = sorted(list(RAW_DIR.glob("*.xls")) + list(RAW_DIR.glob("*.xlsx")))
if not xls_files:
    raise SystemExit(f"[错误] {RAW_DIR} 下没有数据文件，请先运行 01_load.py 放好数据。")
xls = pd.ExcelFile(xls_files[0])
orders = xls.parse("Orders")
people = xls.parse("People")
returns = xls.parse("Returns")
print(f"[读入] Orders {orders.shape} | People {people.shape} | Returns {returns.shape}")

n0 = len(orders)

# ---- 1. 类型校验/兜底 ----
orders["Order Date"] = pd.to_datetime(orders["Order Date"], errors="coerce")
orders["Ship Date"] = pd.to_datetime(orders["Ship Date"], errors="coerce")
for col in ["Sales", "Quantity", "Discount", "Profit"]:
    orders[col] = pd.to_numeric(orders[col], errors="coerce")
bad_dates = orders["Order Date"].isna().sum()
print(f"\n[1 类型] 日期解析失败行数: {bad_dates}（>0 则这些行会在异常步剔除）")

# ---- 2. 缺失值 ----
miss = orders.isna().sum()
miss = miss[miss > 0]
print(f"[2 缺失] 关键字段缺失情况: {'无缺失' if len(miss) == 0 else miss.to_dict()}"
      f"（Postal Code 缺失属正常，保留不删）")

# ---- 3. 异常值 ----
mask_bad = (
    (orders["Quantity"] <= 0)
    | (orders["Sales"] < 0)
    | orders["Order Date"].isna()
)
n_bad = mask_bad.sum()
orders = orders[~mask_bad].copy()
n_neg_profit = (orders["Profit"] < 0).sum()
print(f"[3 异常] 剔除 Quantity<=0 / Sales<0 / 非法日期 共 {n_bad} 行")
print(f"[3 异常] Profit<0 共 {n_neg_profit} 行 —— 属退货/高折扣的正常业务现象，保留不删")

# ---- 4. 重复值 ----
dup_mask = orders.duplicated(subset=["Order ID", "Product ID"], keep="first")
n_dup = dup_mask.sum()
orders = orders[~dup_mask].copy()
print(f"[4 重复] 按 Order ID + Product ID 去重，删除 {n_dup} 行")

# ---- 5. 派生字段 ----
orders["Order Year"] = orders["Order Date"].dt.year
orders["Order Month"] = orders["Order Date"].dt.to_period("M").astype(str)
orders["Profit Margin"] = orders.apply(
    lambda r: r["Profit"] / r["Sales"] if r["Sales"] != 0 else 0, axis=1
)
returned_ids = set(
    returns.loc[returns["Returned"].astype(str).str.strip().str.lower()
                .isin(["yes", "y", "是"]), "Order ID"]
)
orders["Is_Returned"] = orders["Order ID"].isin(returned_ids).astype(int)
n_ret = orders["Is_Returned"].sum()
print(f"[5 派生] 新增 Order Year / Order Month / Profit Margin / Is_Returned")
print(f"[5 派生] 命中退货的订单行数: {n_ret}（Returns 表共 {len(returns)} 个订单号）")

# ---- 6. 多表对齐 ----
orders = orders.merge(
    people.rename(columns={})[["Region", "Regional Manager"]],
    on="Region", how="left"
)
unmatched_region = orders["Regional Manager"].isna().sum()
print(f"[6 对齐] 左连 People（按 Region）补 Regional Manager，未匹配行数: {unmatched_region}")

# ---- 7. 输出 ----
out = CLEAN_DIR / "orders_clean.csv"
orders.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n[输出] {out.name}  {orders.shape[0]} 行 × {orders.shape[1]} 列")
print(f"[对账] 清洗前 {n0} 行 -> 清洗后 {orders.shape[0]} 行（剔除 {n0 - orders.shape[0]} 行）")
print("[下一步] 运行 03_analysis.py 做六个模型分析。")
