# -*- coding: utf-8 -*-
"""
第9章 K-means 用户细分 —— 由《Kmeans 的应用 - 用户细分.ipynb》整理成 .py
======================================================================
银行交易行为数据（柜台 / ATM / POS / 信用卡次数），教材流程：
  相关矩阵 → 标准化 → PCA 看保留几个成分 → 因子分析(top_n=3 + varimax)
  → 因子得分 ATM_POS / TBM / CSC → 偏度检查 → 第一次 KMeans(3)
  → QuantileTransformer 把因子得分转成正态 → 第二次 KMeans(3)
  → 用决策树解释聚类结果。

关于 fa_kit 与 pydotplus：
  - 教材用 fa_kit 做因子提取和 varimax 旋转，这个包装不到 → 用本文件里的
    factor_scores() 实现同一套算法（相关矩阵特征分解 + Kaiser varimax）；
  - 教材用 pydotplus + graphviz 画决策树，需要额外安装 dot 二进制 →
    改用 sklearn.tree.plot_tree（matplotlib 直接出图，无外部依赖）。
  - 教材的 `get_ipython().magic("matplotlib inline")` 是 notebook 专属，去掉。

数据：data/profile_bank.csv（10 万条客户记录）。
运行：整份代码粘进 Jupyter 一个 cell 也能跑；KMeans 设了 random_state 保证结果可复现。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, tree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('profile_bank.csv',)   # 该脚本会用到的 data/ 下的文件或子目录

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


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def varimax(loadings, gamma=1.0, max_iter=100, tol=1e-6):
    """最大方差旋转（Kaiser 迭代），返回旋转后的载荷矩阵。"""
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


def factor_scores(X, n_factors=3):
    """对应 fa_kit 的 load_data_samples + extract_components + top_n + varimax
    + get_component_scores。fa_kit 返回的是白化得分（除以 √特征值），
    这里照做，量纲才能和教材一致；第7章教材用 np.dot(data, 载荷) 是未白化版本。"""
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


def skew_table(df, cols):
    return pd.Series({c: abs(df[c].skew()) for c in cols}).sort_values(ascending=False)


def main():
    model_data = pd.read_csv(os.path.join(DATA, "profile_bank.csv"))
    show("数据规模", model_data.shape)
    raw = model_data.loc[:, "CNT_TBM":"CNT_CSC"]
    show("四个行为字段的相关矩阵（看要不要降维）", raw.corr(method="pearson").round(3))

    X = preprocessing.scale(raw)
    pca = PCA(n_components=3).fit(X)
    show("PCA 前 3 个成分解释的方差", pd.Series(pca.explained_variance_).round(3))
    show("PCA 前 3 个成分解释方差比", pd.Series(pca.explained_variance_ratio_).round(3))

    rot, scores = factor_scores(X, n_factors=3)
    cols = ["ATM_POS", "TBM", "CSC"]
    show("varimax 旋转后的载荷（对应 fa.comps['rot']）",
         pd.DataFrame(rot, index=raw.columns, columns=cols).round(3))
    fa_scores = pd.DataFrame(scores, columns=cols)
    show("因子得分前 5 行", fa_scores.head().round(3))
    show("第一次聚类前：因子得分绝对偏度", skew_table(fa_scores, cols).round(3))

    km1 = KMeans(n_clusters=3, random_state=12345, n_init=10).fit(fa_scores)
    labeled1 = model_data.join(pd.Series(km1.labels_, name="clustor"))
    show("第一次 KMeans 各类样本量", labeled1.clustor.value_counts().sort_index())

    # 偏度大时先做正态变换，再聚类
    qt = preprocessing.QuantileTransformer(output_distribution="normal", random_state=0)
    trans = pd.DataFrame(qt.fit_transform(fa_scores), columns=cols)
    show("正态变换后的绝对偏度", skew_table(trans, cols).round(3))
    km2 = KMeans(n_clusters=3, random_state=12345, n_init=10).fit(trans)
    labeled2 = model_data.join(pd.Series(km2.labels_, name="clustor"))
    show("第二次 KMeans 各类样本量", labeled2.clustor.value_counts().sort_index())
    show("两次聚类各类占比对比", pd.DataFrame({
        "变换前": labeled1.clustor.value_counts(normalize=True).sort_index(),
        "变换后": labeled2.clustor.value_counts(normalize=True).sort_index()}).round(4))

    # 教材的饼图 + 正态化前后对比
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labeled1.clustor.value_counts().sort_index().plot(
        kind="pie", ax=axes[0], autopct="%.1f%%",
        colors=["#5aa9ff", "#8b7cff", "#3ddc97"])
    axes[0].set(title="变换前 KMeans 各类占比", ylabel="")
    pd.DataFrame({"变换前": labeled1.clustor.value_counts().sort_index(),
                  "变换后": labeled2.clustor.value_counts().sort_index()}).plot(
        kind="bar", ax=axes[1], color=["#5aa9ff", "#ff7b7b"])
    axes[1].set(title="两类样本量对比", xlabel="类别", ylabel="客户数")
    axes[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    p1 = os.path.join(CHARTS, "9.1_两次聚类占比对比.png")
    fig.savefig(p1, dpi=120); plt.close(fig)

    # 决策树解释：用原始行为字段解释第二次聚类的标签
    clf = tree.DecisionTreeClassifier(
        criterion="gini", max_depth=3, min_samples_split=100,
        min_samples_leaf=100, random_state=12345)
    clf.fit(model_data, km2.labels_)
    show("决策树特征重要性",
         pd.Series(clf.feature_importances_, index=model_data.columns).round(4))
    show("决策树在训练集上的准确率",
         round(clf.score(model_data, km2.labels_), 4))

    fig, ax = plt.subplots(figsize=(13, 6))
    tree.plot_tree(clf, feature_names=list(model_data.columns),
                   class_names=["0", "1", "2"], filled=True,
                   rounded=True, fontsize=8, ax=ax)
    ax.set_title("用决策树解释 KMeans 分群（max_depth=3）")
    fig.tight_layout()
    p2 = os.path.join(CHARTS, "9.1_分群解释决策树.png")
    fig.savefig(p2, dpi=120); plt.close(fig)
    print(f"\n图已保存：{os.path.basename(p1)}、{os.path.basename(p2)}")


if __name__ == "__main__" or True:
    main()
