# -*- coding: utf-8 -*-
"""
第6章 数据预处理基础 —— 由《数据预处理.ipynb》整理成 .py
========================================================
教材用电信营销数据演示"脏数据识别"：先看分布，发现 AvgIncome 里的 0
其实代表缺失，再把它换成 NaN 后重新看分布；AvgHomeValue 也有同样问题。

对教材代码的修正：`np.NaN` 在 numpy 2.x 已移除 → 改 `np.nan`；
`plt.hist` 遇到 NaN 需要显式 range，教材已这么写，这里保留并补上缺失比例统计。

数据：data/teleco_camp_orig.csv（9686 条营销记录）。
输出：charts/6.2_AvgIncome处理前后对比.png（对应教材两张直方图）。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('teleco_camp_orig.csv',)   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\量化策略分析复现\第6章-市场调研与数据预处理"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '第6章-市场调研与数据预处理'),
              os.path.join(os.getcwd(), '量化策略分析复现', '第6章-市场调研与数据预处理')]
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
    camp = pd.read_csv(os.path.join(DATA, "teleco_camp_orig.csv"))
    show("数据规模", camp.shape)
    show("AvgIncome 描述统计（含大量 0）", camp["AvgIncome"].describe().round(2))
    show("AvgHomeValue 描述统计", camp["AvgHomeValue"].describe().round(2))
    show("两个字段里 0 的个数",
         pd.Series({"AvgIncome": (camp.AvgIncome == 0).sum(),
                    "AvgHomeValue": (camp.AvgHomeValue == 0).sum()}))

    # 教材认为 0 是缺失值标记，直接替换成 NaN（numpy 2 里写 np.nan）
    clean = camp.copy()
    clean["AvgIncome"] = clean["AvgIncome"].replace({0: np.nan})
    show("替换后 AvgIncome 缺失比例", f"{clean.AvgIncome.isna().mean():.2%}")
    show("替换后 AvgIncome 描述统计", clean["AvgIncome"].describe().round(2))

    # 用非 0 的均值填补，作为教材"处理 0 值"的后续一步
    fill_mean = clean["AvgIncome"].mean()
    clean["AvgIncome_filled"] = clean["AvgIncome"].fillna(fill_mean)
    show("均值填补值", round(fill_mean, 2))
    show("填补后缺失比例", f"{clean.AvgIncome_filled.isna().mean():.2%}")

    # 两张教材直方图合并成一张对比图
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(camp["AvgIncome"], bins=20, color="#5aa9ff", edgecolor="white")
    ax[0].set(title="处理前：0 值堆在左侧", xlabel="AvgIncome", ylabel="频数")
    valid = clean["AvgIncome"].dropna()
    ax[1].hist(valid, bins=20, range=(valid.min(), valid.max()),
               color="#3ddc97", edgecolor="white")
    ax[1].set(title="0 换成缺失后", xlabel="AvgIncome", ylabel="频数")
    fig.tight_layout()
    out = os.path.join(CHARTS, "6.2_AvgIncome处理前后对比.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\n图已保存：{out}")


if __name__ == "__main__" or True:
    main()
