# -*- coding: utf-8 -*-
"""
第5章 描述性统计分析与绘图 —— 由《第5章 描述性统计分析与绘图.ipynb》整理成 .py
==========================================================================
教材 3.1 用二手房价格数据讲连续变量的描述统计：集中趋势、离散程度、
分布的偏度峰度，以及 pearson / spearman / kendall 三种相关系数。

数据：data/sndHsPr.csv（16210 条二手房记录，用到 AREA 和 price 两列）。
输出：统计量打印 + charts/5.2_价格分布直方图.png。
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
_EXPECTED = ('sndHsPr.csv',)   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\量化策略分析复现\第5章-数据可视化"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '第5章-数据可视化'),
              os.path.join(os.getcwd(), '量化策略分析复现', '第5章-数据可视化')]
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
    data = pd.read_csv(os.path.join(DATA, "sndHsPr.csv"))
    show("数据规模与字段", f"{data.shape}，字段：{list(data.columns)}")
    show("price / AREA 前 5 行", data[["AREA", "price"]].head())

    # 3.1.3 集中趋势
    show("3.1.3 均值、中位数、标准差", data.price.agg(["mean", "median", "std"]).round(2))
    show("3.1.3 四分位数", data.price.quantile([.25, .5, .75]).round(2))

    # 3.1.4 离散程度
    show("3.1.4 极差 / 方差 / 标准差",
         pd.Series({"极差": data.price.max() - data.price.min(),
                    "方差": data.price.var(), "标准差": data.price.std()}).round(2))

    # 3.1.5 偏度、峰度与相关系数
    np.random.seed(0)
    normal = pd.Series(np.random.randn(1000), name="normal")
    show("3.1.5 标准正态样本的偏度/峰度（理论接近 0）",
         pd.Series({"skew": normal.skew(), "kurtosis": normal.kurtosis()}).round(4))
    show("3.1.5 真实房价的偏度/峰度",
         pd.Series({"skew": data.price.skew(), "kurtosis": data.price.kurtosis()}).round(4))
    for method in ("pearson", "spearman", "kendall"):
        show(f"3.1.5 AREA 与 price 的 {method} 相关系数",
             data[["AREA", "price"]].corr(method).round(4))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(data.price, bins=20, color="#5aa9ff", edgecolor="white")
    ax[0].set(title="price 直方图（教材原图）", xlabel="price", ylabel="频数")
    ax[1].scatter(data.AREA, data.price, s=4, alpha=.15, color="#ff7b7b")
    ax[1].set(title="AREA 与 price 散点", xlabel="AREA", ylabel="price")
    fig.tight_layout()
    out = os.path.join(CHARTS, "5.2_价格分布与面积散点.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\n图已保存：{out}")


if __name__ == "__main__" or True:
    main()
