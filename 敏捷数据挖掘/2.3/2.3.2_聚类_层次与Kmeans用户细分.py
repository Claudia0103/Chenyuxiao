# -*- coding: utf-8 -*-
"""
2.3（下）用户画像之聚类 —— 合并教材 2.3.4 / 2.3.5 两个 notebook
================================================================
  1) 10 城市经济数据：因子得分 → Ward 层次聚类 → 树状图 → 切 2 类 → 簇中心对比；
  2) 银行客户细分（data_clust.csv 16.7 万行）：清洗 → 爱好广度构造 →
     连续变量 QuantileTransformer 正态化 → 按业务分成 household / hobby 两组
     → 各自做因子分析 → 轮廓系数选 K → KMeans 分群 → 交叉透视 → 决策树解释。

替代与修正：
  - fa_kit → 内联「相关矩阵特征分解 + varimax」；
  - pydotplus + graphviz → sklearn.tree 的特征重要性 / plot_tree；
  - `np.NaN` → `np.nan`；`sch.distance.pdist` → scipy.spatial.distance.pdist；
  - 16.7 万行全量跑 KMeans 选 K 很慢，这里抽样 20000 行、选 K 时再抽 3000 行，
    保持方法一致、结果可复现（SAMPLE_ROWS 可改成 None 用全量）。

数据：data/cities_10.csv（GBK）、data/data_clust.csv。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as sch

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('cities_10.csv',)   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\敏捷数据挖掘复现\2.3用户画像"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '2.3用户画像'),
              os.path.join(os.getcwd(), '敏捷数据挖掘复现', '2.3用户画像')]
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

SAMPLE_ROWS = 20000       # 16.7 万行太大；改 None 可跑全量
RANDOM_STATE = 12345


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def save(fig, name):
    p = os.path.join(CHARTS, f"2.3.2_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def varimax(loadings, gamma=1.0, max_iter=100, tol=1e-6):
    p, k = loadings.shape
    rot = np.eye(k)
    old = 0.0
    for _ in range(max_iter):
        L = loadings @ rot
        u, s, vh = np.linalg.svd(
            loadings.T @ (L ** 3 - (gamma / p) * L @ np.diag(np.diag(L.T @ L))))
        rot = u @ vh
        if old and s.sum() / old < 1 + tol:
            break
        old = s.sum()
    return loadings @ rot


def factor_scores(X, n_factors):
    R = np.corrcoef(X, rowvar=False)
    vals, vecs = np.linalg.eigh(R)
    o = np.argsort(vals)[::-1]
    vals, vecs = vals[o], vecs[:, o]
    L = vecs[:, :n_factors] * np.sqrt(vals[:n_factors])
    rot = varimax(L)
    for j in range(n_factors):
        if rot[:, j].sum() < 0:
            rot[:, j] *= -1
    return rot, X @ rot / np.sqrt(vals[:n_factors])


def city_hierarchy():
    cities = pd.read_csv(os.path.join(DATA, "cities_10.csv"), encoding="gbk")
    X = StandardScaler().fit_transform(cities.loc[:, "X1":])
    _, scores = factor_scores(X, 2)
    df = cities.join(pd.DataFrame(scores, columns=["Gross", "Avg"]))
    Z = sch.linkage(pdist(df[["Gross", "Avg"]], metric="euclidean"), method="ward")
    labels = sch.fcluster(Z, t=2, criterion="maxclust").ravel()
    df["cluster"] = labels
    show("城市层次聚类结果", df[["AREA", "Gross", "Avg", "cluster"]].round(3))
    show("两类城市的指标均值对比",
         df.loc[:, "X1":"X9"].groupby(df.cluster).mean().T.round(1))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sch.dendrogram(Z, labels=df.AREA.tolist(), ax=ax, color_threshold=Z[-2, 2])
    ax.set(title="10 城市 Ward 层次聚类", xlabel="城市", ylabel="距离")
    fig.tight_layout()
    save(fig, "城市层次聚类树状图")


def customer_kmeans():
    raw = pd.read_csv(os.path.join(DATA, "data_clust.csv"), skipinitialspace=True)
    show("客户数据规模", raw.shape)
    customer = raw.sample(n=SAMPLE_ROWS, random_state=RANDOM_STATE).reset_index(drop=True) \
        if SAMPLE_ROWS else raw.copy()

    # 缺失：分类变量用众数填补
    for c in customer.columns:
        if customer[c].isna().any():
            customer[c] = customer[c].fillna(customer[c].mode().iloc[0])
    # 错误值：HH_has_children 的 Y/N 转 0/1；阅读爱好里的 '.' 转 0
    customer["HH_has_children"] = customer["HH_has_children"].replace(
        {"N": 0, "Y": 1, np.nan: 0}).astype(int)
    if "interested_reading" in customer:
        customer["interested_reading"] = pd.to_numeric(
            customer["interested_reading"].replace({".": "0"}), errors="coerce").fillna(0)

    interest = ["interested_travel", "computer_owner", "interested_golf",
                "interested_gambling", "interested_reading"]
    interest = [c for c in interest if c in customer.columns]
    customer["interest"] = customer[interest].sum(axis=1) / len(interest)
    customer = customer.drop(columns=interest)
    show("构造的爱好广度指标（前 5）", customer["interest"].head().round(3))

    continuous = [c for c in ["age", "home_value", "risk_score", "interested_sport",
                              "HH_dieting", "auto_member", "HH_grandparent",
                              "HH_head_age", "loan_ratio"] if c in customer.columns]
    categorical = [c for c in ["marital", "interest", "HH_adults_num"]
                   if c in customer.columns]
    binary = ["HH_has_children"]
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal",
                             random_state=0)
    Xc = qt.fit_transform(customer[continuous])
    Xs = StandardScaler().fit_transform(customer[categorical])
    data = pd.DataFrame(np.hstack([Xc, Xs, customer[binary].values]),
                        columns=continuous + categorical + binary)
    show("预处理后建模矩阵", data.shape)

    household = [c for c in ["age", "marital", "HH_adults_num", "home_value",
                             "risk_score", "HH_grandparent", "HH_head_age",
                             "loan_ratio"] if c in data.columns]
    hobby = [c for c in ["HH_dieting", "auto_member", "interest",
                         "interested_sport"] if c in data.columns]
    show("PCA 判断家庭维度保留几个成分",
         np.cumsum(PCA().fit(data[household]).explained_variance_ratio_).round(3))

    rot_hh, sc_hh = factor_scores(data[household].values, 4)
    rot_hb, sc_hb = factor_scores(data[hobby].values, 3)
    show("家庭维度旋转载荷（前 3 个因子）",
         pd.DataFrame(rot_hh[:, :3], index=household,
                      columns=["life_circle", "finance", "HH_size"]).round(3))
    data_hh = pd.DataFrame(StandardScaler().fit_transform(sc_hh),
                           columns=["life_circle", "finance", "HH_size", "risk"])
    data_hb = pd.DataFrame(StandardScaler().fit_transform(sc_hb),
                           columns=["sports", "health", "leisure"])

    # 轮廓系数选 K
    def sil_curve(X, ks=range(2, 8)):
        idx = np.random.RandomState(0).choice(len(X), min(3000, len(X)), replace=False)
        out = []
        for k in ks:
            km = KMeans(n_clusters=k, n_init=5, random_state=RANDOM_STATE)
            lab = km.fit_predict(X[idx])
            out.append({"K": k, "轮廓系数": silhouette_score(X[idx], lab)})
        return pd.DataFrame(out)

    s_hh, s_hb = sil_curve(data_hh.values), sil_curve(data_hb.values)
    show("家庭维度轮廓系数", s_hh.round(4).to_string(index=False))
    show("偏好维度轮廓系数", s_hb.round(4).to_string(index=False))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s_hh.K, s_hh.轮廓系数, "-o", label="household")
    ax.plot(s_hb.K, s_hb.轮廓系数, "-s", label="hobby")
    ax.set(title="轮廓系数选 K", xlabel="K", ylabel="silhouette")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    save(fig, "轮廓系数选K")

    k_hh = int(s_hh.loc[s_hh.轮廓系数.idxmax(), "K"])
    k_hb = int(s_hb.loc[s_hb.轮廓系数.idxmax(), "K"])
    lab_hh = KMeans(n_clusters=k_hh, n_init=10,
                    random_state=RANDOM_STATE).fit_predict(data_hh.values)
    lab_hb = KMeans(n_clusters=k_hb, n_init=10,
                    random_state=RANDOM_STATE).fit_predict(data_hb.values)
    clusters = customer.join(pd.DataFrame({"hh": lab_hh, "hb": lab_hb}))
    show("分群规模（hh × hb）", pd.crosstab(clusters.hh, clusters.hb))
    pt = pd.pivot_table(clusters, index="hh", columns="hb",
                        values="home_value", aggfunc="mean")
    show("交叉透视：各客群平均房产价值", pt.round(1))

    # 决策树解释聚类标签（教材用 pydotplus，这里用特征重要性，无需 graphviz）
    for name, cols, target in [("hh", household, "hh"), ("hb", hobby, "hb")]:
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200,
                                     random_state=RANDOM_STATE)
        clf.fit(clusters[cols], clusters[target])
        show(f"{name} 分群的解释：特征重要性",
             pd.Series(clf.feature_importances_, index=cols).round(3))

    fig, ax = plt.subplots(figsize=(7, 4))
    pd.crosstab(clusters.hh, clusters.hb).plot(kind="bar", ax=ax,
                                                color=["#5aa9ff", "#8b7cff", "#3ddc97"])
    ax.set(title="家庭 × 偏好 客群规模", xlabel="家庭维度簇", ylabel="客户数")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    save(fig, "客群规模")


def main():
    print("2.3（下）层次聚类与 KMeans 用户细分")
    city_hierarchy()
    customer_kmeans()


if __name__ == "__main__" or True:
    main()
