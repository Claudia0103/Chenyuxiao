# -*- coding: utf-8 -*-
"""
第5章 数据可视化 —— 由《第5章 python数据可视化.ipynb》整理成 .py
================================================================
教材 2.1 是 matplotlib 基础图（线图/饼图/条形图/直方图/散点图/箱线图），
2.2 是 seaborn 统计图（带 KDE 的直方图、二元分布 4 种、热图）。
原 notebook 用 plt.show() 在 Jupyter 里显示，这里统一保存成 PNG 到 charts/，
共 12 张图；每张图前打印用到的数据，方便对照教材。

数据：data/2017股价.xlsx（15 个交易日的开收高低价）。
运行：整份代码粘进 Jupyter 一个 cell 也能跑；__file__ 不存在时按当前目录找 data/。

老 API 修正：matplotlib 3.9 起 plt.boxplot 的 labels 参数改名 tick_labels。
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.sans-serif"] = ["SimHei"]

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('2017股价.xlsx',)   # 该脚本会用到的 data/ 下的文件或子目录

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

_n = [0]


def save(name, grid=None):
    """保存当前图。seaborn 的 figure-level 图传 grid 进来更稳。"""
    _n[0] += 1
    path = os.path.join(CHARTS, f"5.1_{_n[0]:02d}_{name}.png")
    fig = None
    if grid is not None:
        fig = getattr(grid, "figure", None) or getattr(grid, "fig", None)
    (fig or plt).savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    print(f"  图已保存：{os.path.basename(path)}")


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def stock_data():
    return pd.read_excel(os.path.join(DATA, "2017股价.xlsx"))


# ---------------------------------------------------------------- 2.1 matplotlib
def mpl_line():
    df = stock_data()
    show("2.1.1 股价数据", df[["日期", "收盘价", "开盘价", "最高价", "最低价"]].head())
    plt.figure(figsize=(10, 4))
    for col, marker in [("收盘价", "-*"), ("开盘价", "-o"),
                        ("最高价", "-v"), ("最低价", "-^")]:
        plt.plot(range(15), df.loc[:, col], marker)
    plt.title("股价走势图")
    plt.xlabel("时间"); plt.ylabel("股价")
    plt.xticks(range(15), df["日期"], rotation=30)
    plt.legend(["收盘价", "开盘价", "最高价", "最低价"])
    plt.grid(alpha=.3)
    save("股价走势图")


def mpl_pie():
    np.random.seed(1)
    df = pd.DataFrame({"方案": ["plan A", "plan B", "plan C", "plan D", "plan E"],
                       "投票人数": np.random.randint(10, 100, 5)})
    show("2.1.2 投票数据", df)
    plt.figure(figsize=(6, 6))
    plt.pie(df.投票人数, labels=df.方案, autopct="%.2f%%",
            textprops={"fontsize": 13})
    plt.title("方案投票占比饼图", fontsize=14)
    save("方案投票占比饼图")


def mpl_bar():
    np.random.seed(1)
    df = pd.DataFrame({"方案": ["plan A", "plan B", "plan C", "plan D", "plan E"],
                       "投票人数": np.random.randint(10, 100, 5)})
    plt.figure(figsize=(8, 4))
    plt.bar(df.方案, df.投票人数, width=.6, color="#5aa9ff")
    plt.title("各方案投票人数柱形图", fontsize=14)
    plt.xlabel("方案", fontsize=13); plt.ylabel("投票人数", fontsize=13)
    save("柱形图")

    s = df.sort_values("投票人数")
    plt.figure(figsize=(8, 4))
    plt.barh(s.方案, s.投票人数, height=.6, color="#8b7cff")
    plt.title("各方案投票人数条形图", fontsize=14)
    plt.xlabel("投票人数", fontsize=13); plt.ylabel("方案", fontsize=13)
    save("水平条形图")


def mpl_hist():
    np.random.seed(1)
    data = np.random.normal(loc=80, scale=20, size=200).astype(int)
    show("2.1.4 直方图数据前 10 个", data[:10])
    bins = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins, color="#3ddc97", edgecolor="white")
    plt.xticks(bins)
    plt.title("成绩分布直方图")
    save("直方图")


def mpl_scatter():
    df = pd.DataFrame({"产量": [2200, 3300, 4400, 5500, 5700, 7300, 8000],
                       "降水量": [22, 39, 55, 66, 105, 100, 122]})
    show("2.1.5 产量与降水量", df)
    plt.figure(figsize=(6, 4))
    plt.scatter(df.降水量, df.产量, s=60, color="#ff7b7b")
    plt.xlabel("降水量"); plt.ylabel("产量")
    plt.title("降水量与产量散点图")
    plt.grid(alpha=.3)
    save("散点图")


def mpl_box():
    np.random.seed(1)
    df = pd.DataFrame({"语文": np.random.randint(60, 150, 10),
                       "英语": np.random.randint(40, 150, 10),
                       "数学": np.random.randint(20, 150, 10)})
    show("2.1.6 三科成绩", df)
    plt.figure(figsize=(8, 6))
    plt.boxplot([df.语文, df.英语, df.数学], tick_labels=["语文", "英语", "数学"])
    plt.ylabel("分数")
    plt.title("三科成绩箱线图")
    save("箱线图")


# ---------------------------------------------------------------- 2.2 seaborn
def sns_dist():
    np.random.seed(1)
    data = np.random.normal(loc=80, scale=20, size=200).astype(int)
    g = sns.displot(data, bins=[30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
                    kde=True, color="#5aa9ff")
    g.set_axis_labels("取值", "频数")
    save("带核密度估计的直方图", g)


def sns_joint():
    np.random.seed(0)
    mean, cov = [0, 1], [(1, .5), (.5, 1)]
    df = pd.DataFrame(np.random.multivariate_normal(mean, cov, 200),
                      columns=["x", "y"])
    show("2.2.2 二元正态样本", df.head())

    g = sns.jointplot(x="x", y="y", data=df)
    save("二元分布_散点加边缘直方图", g)
    g = sns.relplot(x="x", y="y", data=df)
    save("二元分布_relplot", g)
    g = sns.jointplot(x="x", y="y", data=df, kind="kde")
    save("二元分布_核密度", g)
    g = sns.displot(data=df, x="x", y="y", kind="kde")
    save("二元分布_二维核密度", g)


def sns_heatmap():
    np.random.seed(0)
    df0 = pd.DataFrame(np.random.uniform(-1, 1, (7, 7)),
                       index=list("abcdefg"), columns=list("ABCDEFG"))
    show("2.2.3 相关矩阵样式数据", df0.round(2))
    sns.set(font_scale=1.5)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df0, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("7×7 随机矩阵热图")
    save("热图")
    sns.set_theme()   # 恢复默认，避免影响后面的图


def main():
    print("第5章 python 数据可视化 —— matplotlib 6 类 + seaborn 6 类")
    mpl_line(); mpl_pie(); mpl_bar(); mpl_hist(); mpl_scatter(); mpl_box()
    sns_dist(); sns_joint(); sns_heatmap()
    print(f"\n共保存 {_n[0]} 张图到 {CHARTS}")


if __name__ == "__main__" or True:
    main()
