# -*- coding: utf-8 -*-
"""
第5章 案例2：Vintage 报表 —— 由 Vintage.ipynb / 主 notebook 5.7 整理
=====================================================================
教材用 pymysql + sqlalchemy 连 `mysql://root@.../risk` 跑一段按放款月汇总
mob1~mob4 的 SQL。本机没有那个 MySQL 库，但教材把三张表的 Navicat 导出
（含 CREATE TABLE 和 15841 条 INSERT）一起给了，所以这里直接解析 .sql 文件
还原成 DataFrame，再用 pandas 算 Vintage，效果等价且不需要数据库。

产出：
  - 解析 customer_detail / repayment_sum_month / daily_report 三张表；
  - 按放款月 × 账龄(MOB) 计算逾期率 = Σ逾期本金 / Σ应还金额；
  - 教材 SQL 的放款月汇总（放款金额、mob1~mob4）等价结果；
  - Vintage 热力图与各放款月的 MOB 曲线。

数据：data/Vintage/sql/*.sql。运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
CHAPTER_DIR = ""
_EXPECTED = ("Vintage/sql/repayment_sum_month.sql",)


def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\金融数据分析案例实战复现\第5章-A卡评分卡"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(), os.path.join(os.getcwd(), "第5章-A卡评分卡"),
              os.path.join(os.getcwd(), "金融数据分析案例实战复现", "第5章-A卡评分卡")]
    for r in roots:
        d = os.path.join(r, "data")
        if os.path.isdir(d) and any(os.path.exists(os.path.join(d, x)) for x in _EXPECTED):
            return r
    return os.getcwd()


_HERE = _locate_chapter_dir()
DATA = os.path.join(_HERE, "data")
CHARTS = os.path.join(_HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def save(fig, name):
    p = os.path.join(CHARTS, f"5.2_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def parse_sql_dump(path):
    """解析 Navicat 导出的 CREATE TABLE + INSERT INTO ... VALUES(...);"""
    text = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"CREATE TABLE `?\w+`?\s*\((.*?)\)\s*ENGINE", text, re.S)
    if not m:
        raise ValueError(f"没找到 CREATE TABLE: {path}")
    cols = re.findall(r"`([^`]+)`\s+[a-zA-Z]", m.group(1))
    rows = []
    for ins in re.finditer(r"INSERT INTO `?\w+`? VALUES\s*(.*?);", text, re.S | re.I):
        for tup in re.findall(r"\((.*?)\)", ins.group(1), re.S):
            vals = []
            for tok in re.findall(r"'[^']*'|[^,]+", tup):
                v = tok.strip()
                if not v or v.upper() == "NULL":
                    vals.append(None)
                else:
                    vals.append(v.strip("'"))
            if len(vals) == len(cols):
                rows.append(vals)
            else:
                # 长文本列里可能含逗号，长度不符时只保留能对应上的前几列
                if len(vals) > len(cols):
                    rows.append(vals[:len(cols)])
    return pd.DataFrame(rows, columns=cols)


def month_diff(a, b):
    """观测月 - 放款月，返回相差月数（MOB）。"""
    ya, ma = int(str(a)[:4]), int(str(a)[5:7])
    yb, mb = int(str(b)[:4]), int(str(b)[5:7])
    return (ya - yb) * 12 + (ma - mb)


def main():
    sql_dir = os.path.join(DATA, "Vintage", "sql")
    customer = parse_sql_dump(os.path.join(sql_dir, "customer_detail.sql"))
    repay = parse_sql_dump(os.path.join(sql_dir, "repayment_sum_month.sql"))
    daily = parse_sql_dump(os.path.join(sql_dir, "daily_report.sql"))
    show("三张表规模", {"customer_detail": customer.shape,
                      "repayment_sum_month": repay.shape,
                      "daily_report": daily.shape})
    show("repayment 字段", list(repay.columns))
    show("customer_detail 前 3 行", customer.head(3).to_string(index=False))

    for c in ["应还金额", "剩余本金", "逾期本金", "分期数", "剩余期数", "逾期期数"]:
        repay[c] = pd.to_numeric(repay[c], errors="coerce").fillna(0)
    customer["合同金额"] = pd.to_numeric(customer["合同金额"], errors="coerce").fillna(0)
    customer["放款月"] = customer["放款日期"].astype(str).str[:7]

    # 教材 SQL 的按放款月汇总：放款金额 + mob1~mob4 逾期率
    amount = customer.groupby("放款月")["合同金额"].sum() / 10000
    repay["MOB"] = [month_diff(o, l) for o, l in
                    zip(repay["观测月"], repay["放款月"])]
    vintage = (repay[repay.MOB.between(0, 8)]
               .groupby(["放款月", "MOB"])
               .agg(应还=("应还金额", "sum"), 逾期=("逾期本金", "sum"),
                    合同数=("合同ID", "nunique"))
               .reset_index())
    vintage["逾期率"] = vintage["逾期"] / vintage["应还"].replace(0, np.nan)
    matrix = vintage.pivot(index="放款月", columns="MOB", values="逾期率")
    matrix = matrix.reindex(columns=range(1, 7))
    show("Vintage 矩阵（行=放款月，列=MOB，值=逾期本金/应还金额）",
         matrix.round(4).head(12))

    report = pd.DataFrame({"放款金额(万元)": amount})
    for mob in range(1, 5):
        col = vintage[vintage.MOB == mob].groupby("放款月")["逾期率"].mean()
        report[f"mob{mob}"] = col
    report = report.dropna(how="all", subset=[f"mob{i}" for i in range(1, 5)])
    show("教材 SQL 等价汇总（前 10 个放款月）", report.round(4).head(10))

    fig, ax = plt.subplots(figsize=(9, 5))
    data = matrix.loc[matrix.index >= "2017-01"].head(18)
    im = ax.imshow(data.values.astype(float), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(data.shape[1])); ax.set_xticklabels([f"MOB{i+1}" for i in range(data.shape[1])])
    ax.set_yticks(range(len(data))); ax.set_yticklabels(data.index, fontsize=8)
    ax.set_title("Vintage 逾期率热力图（SQL 导出表等价复现）")
    fig.colorbar(im, ax=ax, label="逾期率")
    fig.tight_layout()
    save(fig, "Vintage热力图")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for month in data.index[:6]:
        ax.plot(range(1, 7), data.loc[month].values.astype(float), "-o", label=str(month))
    ax.set(title="各放款月的 MOB 逾期率曲线", xlabel="MOB", ylabel="逾期率")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    save(fig, "MOB曲线")


if __name__ == "__main__" or True:
    main()
