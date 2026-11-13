# -*- coding: utf-8 -*-
"""
第3章 多维特征的客户细分 —— 由《第3章多维特征的客户细分代码通过.ipynb》整理
==========================================================================
流程和三级教材 2.3.5 同源（同一份 data_clust.csv）：
  缺失填补 → 错误值修订 → 离散变量卡方检验 → 爱好广度构造 →
  连续变量 QuantileTransformer 正态化 + 有序变量标准化 →
  按业务分 household / hobby 两组做因子分析 → 轮廓系数选 K →
  KMeans 双维度分群 → 交叉透视 → 决策树解释簇特征。

替代与修正：
  - fa_kit 装不到 → 内联「相关矩阵特征分解 + Kaiser varimax」；教材用
    `np.dot(标准化数据, 旋转载荷)` 得分，这里保持一致（未白化）；
  - pydotplus + graphviz → sklearn.tree 的特征重要性与 plot_tree；
  - `np.NaN` → `np.nan`；`%matplotlib inline` 去掉；
  - 16.7 万行默认抽 2 万行（data_clust_sample.csv），SAMPLE_ROWS 可改 None 跑全量。

数据：data/data_clust_sample.csv。运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import Binarizer, QuantileTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
CHAPTER_DIR = ""
_EXPECTED = ("data_clust_sample.csv",)


def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\金融数据分析案例实战复现\第3章-客户细分"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(), os.path.join(os.getcwd(), "第3章-客户细分"),
              os.path.join(os.getcwd(), "金融数据分析案例实战复现", "第3章-客户细分")]
    for r in roots:
        d = os.path.join(r, "data")
        if os.path.isdir(d) and any(os.path.exists(os.path.join(d, x)) for x in _EXPECTED):
            return r
    return os.getcwd()


_HERE = _locate_chapter_dir()
DATA = os.path.join(_HERE, "data")
CHARTS = os.path.join(_HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)

SAMPLE_ROWS = None      # 数据已是 2 万行抽样；改成整数可再抽
RS = 12345


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def save(fig, name):
    p = os.path.join(CHARTS, f"3.1_{name}.png")
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
    """主成分法载荷 + varimax 旋转 + 教材的 np.dot(X, 载荷) 得分。"""
    R = np.corrcoef(X, rowvar=False)
    vals, vecs = np.linalg.eigh(R)
    o = np.argsort(vals)[::-1]
    vals, vecs = vals[o], vecs[:, o]
    L = vecs[:, :n_factors] * np.sqrt(vals[:n_factors])
    rot = varimax(L)
    for j in range(n_factors):
        if rot[:, j].sum() < 0:
            rot[:, j] *= -1
    return rot, X @ rot


def main():
    bank = pd.read_csv(os.path.join(DATA, "data_clust_sample.csv"), skipinitialspace=True)
    if SAMPLE_ROWS:
        bank = bank.sample(n=SAMPLE_ROWS, random_state=RS).reset_index(drop=True)
    show("客户数据规模", bank.shape)
    show("有缺失的列", [c for c in bank.columns if bank[c].isna().any()])

    for c in bank.columns:
        if bank[c].isna().any():
            bank[c] = bank[c].fillna(bank[c].mode().iloc[0])
    bank["HH_has_children"] = (bank["HH_has_children"]
                               .replace({"N": 0, "Y": 1, np.nan: 0}).astype(int))
    bank["interested_reading"] = pd.to_numeric(
        bank["interested_reading"].replace({".": "0"}), errors="coerce").fillna(0)
    show("填补后缺失合计", int(bank.isna().sum().sum()))

    cat_cols = ["interested_travel", "computer_owner", "marital",
                "interested_golf", "interested_gambling", "HH_has_children",
                "interested_reading"]
    sample = bank[cat_cols].sample(min(3000, len(bank)), random_state=RS)
    indep = []
    for a, b in combinations(cat_cols, 2):
        tab = pd.crosstab(sample[a], sample[b])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        p = stats.chi2_contingency(tab)[1]
        if p > .05:
            indep.append((a, b, round(float(p), 4)))
    show("卡方检验：不显著相关的离散变量对（前 8）", indep[:8])

    bank["interested_reading"] = Binarizer(threshold=1.5).fit_transform(
        bank[["interested_reading"]]).astype(int)
    interest = ["interested_travel", "computer_owner", "interested_golf",
                "interested_gambling", "interested_reading"]
    bank["interest"] = bank[interest].sum(axis=1) / len(interest)
    bank = bank.drop(columns=interest)
    show("爱好广度指标（前 5）", bank.interest.head().round(3))

    continuous = ["age", "home_value", "risk_score", "interested_sport",
                  "HH_dieting", "auto_member", "HH_grandparent",
                  "HH_head_age", "loan_ratio"]
    categorical = ["marital", "interest", "HH_adults_num"]
    binary = ["HH_has_children"]
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal",
                             random_state=0)
    Xc = qt.fit_transform(bank[continuous])
    Xs = StandardScaler().fit_transform(bank[categorical])
    data = pd.DataFrame(np.hstack([Xc, Xs, bank[binary].values]),
                        columns=continuous + categorical + binary)
    show("建模矩阵", data.shape)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(bank[continuous[0]], bins=30, color="#5aa9ff")
    ax[0].set_title("正态化前：age")
    ax[1].hist(Xc[:, 0], bins=30, color="#3ddc97")
    ax[1].set_title("QuantileTransformer 后")
    fig.tight_layout()
    save(fig, "正态化前后")

    household = ["age", "marital", "HH_adults_num", "home_value",
                 "risk_score", "HH_grandparent", "HH_head_age", "loan_ratio"]
    hobby = ["HH_dieting", "auto_member", "interest", "interested_sport"]
    show("household 维度 PCA 累计解释方差比",
         np.cumsum(PCA().fit(data[household]).explained_variance_ratio_).round(3))

    rot_hh, sc_hh = factor_scores(data[household].values, 4)
    rot_hb, sc_hb = factor_scores(data[hobby].values, 3)
    show("household 旋转载荷（前 3 个因子）",
         pd.DataFrame(rot_hh[:, :3], index=household,
                      columns=["life_circle", "finance", "HH_size"]).round(3))
    show("hobby 旋转载荷",
         pd.DataFrame(rot_hb, index=hobby,
                      columns=["sports", "health", "leisure"]).round(3))
    data_hh = pd.DataFrame(StandardScaler().fit_transform(sc_hh),
                           columns=["life_circle", "finance", "HH_size", "risk"])
    data_hb = pd.DataFrame(StandardScaler().fit_transform(sc_hb),
                           columns=["sports", "health", "leisure"])

    def sil_curve(X, ks=range(2, 9)):
        idx = np.random.RandomState(0).choice(len(X), min(3000, len(X)), replace=False)
        return pd.DataFrame([
            {"K": k, "轮廓系数": silhouette_score(
                X[idx], KMeans(n_clusters=k, n_init=5, random_state=RS)
                .fit_predict(X[idx]))}
            for k in ks])

    s_hh, s_hb = sil_curve(data_hh.values), sil_curve(data_hb.values)
    show("household 轮廓系数", s_hh.round(4).to_string(index=False))
    show("hobby 轮廓系数", s_hb.round(4).to_string(index=False))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s_hh.K, s_hh.轮廓系数, "-o", label="household")
    ax.plot(s_hb.K, s_hb.轮廓系数, "-s", label="hobby")
    ax.set(title="轮廓系数选 K", xlabel="K", ylabel="silhouette")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    save(fig, "轮廓系数选K")

    k_hh = int(s_hh.loc[s_hh.轮廓系数.idxmax(), "K"])
    k_hb = int(s_hb.loc[s_hb.轮廓系数.idxmax(), "K"])
    show("选定的 K", {"household": k_hh, "hobby": k_hb})
    lab_hh = KMeans(n_clusters=k_hh, n_init=10, random_state=RS).fit_predict(data_hh.values)
    lab_hb = KMeans(n_clusters=k_hb, n_init=10, random_state=RS).fit_predict(data_hb.values)
    clusters = bank.join(pd.DataFrame({"hh": lab_hh, "hb": lab_hb}))
    show("客群规模（hh × hb）", pd.crosstab(clusters.hh, clusters.hb))
    show("交叉透视：各客群平均房产价值",
         pd.pivot_table(clusters, index="hh", columns="hb",
                        values="home_value", aggfunc="mean").round(1))

    for name, cols, target in [("hh", household, "hh"), ("hb", hobby, "hb")]:
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200,
                                     random_state=RS)
        clf.fit(clusters[cols], clusters[target])
        show(f"{name} 分群解释：特征重要性",
             pd.Series(clf.feature_importances_, index=cols).round(3))
        fig, ax = plt.subplots(figsize=(11, 4.5))
        plot_tree(clf, feature_names=cols, filled=True, rounded=True,
                  fontsize=8, max_depth=3, ax=ax)
        ax.set_title(f"{name} 分群决策树（教材 pydotplus 图的等价替代）")
        fig.tight_layout()
        save(fig, f"{name}分群决策树")

    fig, ax = plt.subplots(figsize=(7, 4))
    pd.crosstab(clusters.hh, clusters.hb).plot(kind="bar", ax=ax,
                                                color=["#5aa9ff", "#8b7cff", "#3ddc97"])
    ax.set(title="家庭 × 偏好 客群规模", xlabel="家庭维度簇", ylabel="客户数")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    save(fig, "客群规模")


if __name__ == "__main__" or True:
    main()
