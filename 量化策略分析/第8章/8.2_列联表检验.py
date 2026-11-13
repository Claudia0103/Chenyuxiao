# -*- coding: utf-8 -*-
"""
第8章 列联表检验 —— 由《列联表分析.ipynb》整理成 .py
======================================================
数据：accepts.csv（5845 条贷款申请），检验"是否曾破产 bankruptcy_ind"与
"是否坏客户 bad_ind"是否独立：交叉表 → 行百分比 → 卡方独立性检验。

教材 notebook 的 C2 里那行 `%stats.chi2_contingency(...)` 其实不是魔法命令，
是字符串格式化 `%` 紧跟函数调用，前面多了个空格看起来像 Jupyter magic。
整理成 .py 时用 f-string 写清楚。

另外：bankruptcy_ind 有 217 个缺失，pd.crosstab 默认不计入，
这里把缺失单独列出来说明，避免误以为样本量少了。

运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('accepts.csv',)   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\量化策略分析复现\第8章-统计检验与预测"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '第8章-统计检验与预测'),
              os.path.join(os.getcwd(), '量化策略分析复现', '第8章-统计检验与预测')]
    for r in roots:
        d = os.path.join(r, "data")
        if not os.path.isdir(d):
            continue
        if _EXPECTED and not any(os.path.exists(os.path.join(d, x)) or
                                 any(os.path.exists(os.path.join(d, s, x))
                                     for s in os.listdir(d)
                                     if os.path.isdir(os.path.join(d, s)))
                                 for x in _EXPECTED):
            continue
        return r
    return os.getcwd()

_HERE = _locate_chapter_dir()
DATA = os.path.join(_HERE, "data")
CHARTS = os.path.join(_HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def main():
    accepts = pd.read_csv(os.path.join(DATA, "accepts.csv"))
    show("数据规模与缺失", pd.Series({
        "总行数": len(accepts),
        "bankruptcy_ind 缺失": accepts.bankruptcy_ind.isna().sum(),
        "bad_ind 缺失": accepts.bad_ind.isna().sum()}))

    cross = pd.crosstab(accepts.bankruptcy_ind, accepts.bad_ind, margins=True)
    show("列联表（频数，含合计）", cross)
    row_pct = cross.div(cross["All"], axis=0)
    show("行百分比", row_pct.round(4))

    # 卡方检验只用 2×2 主体，不能把 All 合计行/列算进去
    table = cross.iloc[:2, :2]
    chi2, p, dof, expected = stats.chi2_contingency(table)
    print("\n" + "=" * 10 + " 卡方独立性检验 " + "=" * 10)
    print(f"chisq = {chi2:6.4f}\np-value = {p:6.4f}\ndof = {dof}")
    show("期望频数", pd.DataFrame(expected, index=table.index,
                                columns=table.columns).round(2))
    alpha = 0.05
    print(f"\n结论：p {'<' if p < alpha else '>='} {alpha}，"
          f"{'拒绝独立假设，破产记录与坏客户显著相关' if p < alpha else '不能拒绝独立假设'}")

    fig, ax = plt.subplots(figsize=(7, 4))
    pct = row_pct.iloc[:2, :2] * 100
    pct.plot(kind="bar", stacked=True, ax=ax, color=["#5aa9ff", "#ff7b7b"])
    ax.set(title="曾破产记录与坏客户比例", xlabel="bankruptcy_ind",
           ylabel="行内占比 (%)", ylim=(0, 100))
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="bad_ind")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f%%", label_type="center", fontsize=9)
    fig.tight_layout()
    out = os.path.join(CHARTS, "8.2_破产与坏客户列联表.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"图已保存：{out}")


if __name__ == "__main__" or True:
    main()
