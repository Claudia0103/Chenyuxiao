# -*- coding: utf-8 -*-
"""
第9章 层次聚类 —— 由《hc_city10.ipynb》整理成 .py
=================================================
10 个城市、9 个经济指标：标准化 → PCA 判断成分数 → 因子分析(2 因子 + varimax)
→ 因子得分 Gross/Avg → 城市散点 → Ward 层次聚类 → 树状图。

教材 notebook 里很多格子写着"你的代码"，是留给学员填空的练习，这里全部补上。
fa_kit 同样用自包含的 varimax 实现替换（见 factor_scores）；
教材把中间结果写到 fa_cities_10.csv，这里写到 charts/ 下，不覆盖教材附带文件。

数据：data/cities_10.csv（GBK）、data/fa_cities_10.csv（教材给的因子得分，用来对照）。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as sch

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('cities_10.csv', 'fa_cities_10.csv')   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\量化策略分析复现\第9章-用户分群"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '第9章-用户分群'),
              os.path.join(os.getcwd(), '量化策略分析复现', '第9章-用户分群')]
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

LABELS = {
    "X1": "GDP", "X2": "人均GDP", "X3": "工业增加值", "X4": "第三产业增加值",
    "X5": "固定资产投资", "X6": "基本建设投资", "X7": "社零总额",
    "X8": "海关出口总额", "X9": "地方财政收入",
}


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def varimax(loadings, gamma=1.0, max_iter=100, tol=1e-6):
    p, k = loadings.shape
    rotation = np.eye(k)
    old = 0.0
    for _ in range(max_iter):
        rotated = loadings @ rotation
        u, s, vh = np.linalg.svd(
            loadings.T @ (rotated ** 3 - (gamma / p) * rotated @
                          np.diag(np.diag(rotated.T @ rotated))))
        rotation = u @ vh
        if old != 0 and s.sum() / old < 1 + tol:
            break
        old = s.sum()
    return loadings @ rotation


def factor_scores(X, n_factors=2):
    """对应 fa_kit 的 top_n + varimax + get_component_scores。

    fa_kit 返回的是白化得分（除以 √特征值）：用这个口径算出来的 Gross 与教材
    附带的 fa_cities_10.csv 相关系数 1.0、Avg 约 0.94，量纲一致。
    """
    R = np.corrcoef(X, rowvar=False)
    vals, vecs = np.linalg.eigh(R)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    loadings = vecs[:, :n_factors] * np.sqrt(vals[:n_factors])
    rot = varimax(loadings)
    for j in range(n_factors):
        if rot[:, j].sum() < 0:
            rot[:, j] *= -1
    return rot, X @ rot / np.sqrt(vals[:n_factors])


def main():
    df = pd.read_csv(os.path.join(DATA, "cities_10.csv"), encoding="gbk")
    features = df.loc[:, "X1":]
    show("10 城市 × 9 指标", df.shape)
    show("相关系数矩阵（判断降维必要）", features.corr().round(3))

    X = StandardScaler().fit_transform(features)
    pca = PCA(n_components=3).fit(X)
    show("前 3 个主成分解释的变异", pd.Series(pca.explained_variance_).round(3))
    show("前 3 个主成分解释变异的比例", pd.Series(pca.explained_variance_ratio_).round(4))
    print("  前两个成分已解释绝大部分信息，取 2 个因子")

    rot, scores = factor_scores(X, n_factors=2)
    names = [f"{c} {LABELS.get(c, c)}" for c in features.columns]
    show("旋转后因子载荷", pd.DataFrame(rot, index=names, columns=["Gross", "Avg"]).round(3))

    fa_scores = pd.DataFrame(scores, columns=["Gross", "Avg"])
    cities = df.join(fa_scores)
    show("拼接因子得分后的城市表", cities[["AREA", "Gross", "Avg"]].round(3))
    out_csv = os.path.join(CHARTS, "fa_cities_10_复现.csv")
    cities.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"  中间结果写到：{os.path.basename(out_csv)}")

    # 教材附带的因子得分文件，用来对照方向（符号可能整体相反，属正常）
    ref = pd.read_csv(os.path.join(DATA, "fa_cities_10.csv"))
    show("教材给的 fa_cities_10.csv（前 3 行）", ref[["AREA", "Gross", "Avg"]].head(3).round(3))

    # 城市散点，带城市名标签
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cities.Gross, cities.Avg, s=60, color="#8b7cff")
    for x, y, label in zip(cities.Gross, cities.Avg, cities.AREA):
        ax.text(x, y + .12, str(label), ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="gray", lw=.8); ax.axvline(0, color="gray", lw=.8)
    ax.set(title="10 城市因子得分散点（聚类前）", xlabel="Gross", ylabel="Avg")
    ax.grid(alpha=.3)
    fig.tight_layout()
    p1 = os.path.join(CHARTS, "9.2_城市因子得分散点.png")
    fig.savefig(p1, dpi=120); plt.close(fig)

    # Ward 层次聚类
    dist = pdist(cities[["Gross", "Avg"]], metric="euclidean")
    Z = sch.linkage(dist, method="ward")
    show("linkage 矩阵（每行：两类合并、距离、类内样本数）",
         pd.DataFrame(Z, columns=["类1", "类2", "距离", "样本数"]).round(3))

    fig, ax = plt.subplots(figsize=(10, 5))
    sch.dendrogram(Z, labels=cities.AREA.tolist(), ax=ax,
                   color_threshold=Z[-2, 2])
    ax.set(title="10 城市 Ward 层次聚类树状图", xlabel="城市", ylabel="距离")
    fig.tight_layout()
    p2 = os.path.join(CHARTS, "9.2_层次聚类树状图.png")
    fig.savefig(p2, dpi=120); plt.close(fig)

    # 教材只画树状图，这里补一个切成 3 类的结果，方便看分群
    labels = sch.fcluster(Z, t=3, criterion="maxclust")
    show("切成 3 类的城市归属",
         pd.DataFrame({"城市": cities.AREA, "类别": labels})
         .sort_values("类别").to_string(index=False))
    print(f"\n图已保存：{os.path.basename(p1)}、{os.path.basename(p2)}")


if __name__ == "__main__" or True:
    main()
