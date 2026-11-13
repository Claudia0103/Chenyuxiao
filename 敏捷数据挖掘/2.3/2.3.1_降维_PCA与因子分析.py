# -*- coding: utf-8 -*-
"""
2.3（上）用户画像之降维 —— 合并教材 2.3.1 / 2.3.1-2.3.2 / 2.3.3 三个 notebook
==========================================================================
三个案例：
  1) Loan_aply.csv：5 个贷款评分指标 → PCA 打分（教材用 np.dot 手算第一个主成分）；
  2) cities_10.csv：10 城市 9 经济指标 → 标准化 → PCA → SparsePCA 选稀疏度
     → 因子分析(top_n=2 + varimax) → 因子得分 Gross / Avg；
  3) 变量聚类：教材用 myscript/varSelec.py 的 SparsePCA 挑代表变量，
     该脚本有 best_alpha 未定义的隐患，这里改用更稳的等价做法：
     按 1-|corr| 距离对变量做层次聚类，每簇挑一个与其他变量最相关的代表。

fa_kit 依旧装不到，用本文件内联的「相关矩阵特征分解 + Kaiser varimax」替代；
get_component_scores 是白化得分（除以 √λ），与教材 cities_FA_score.csv 对齐。

数据：data/Loan_aply.csv、data/cities_10.csv（GBK）、data/cities_FA_score.csv（教材答案）。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, SparsePCA
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as sch

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('Loan_aply.csv', 'cities_10.csv', 'cities_FA_score.csv')   # 该脚本会用到的 data/ 下的文件或子目录

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


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def save(fig, name):
    p = os.path.join(CHARTS, f"2.3.1_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def varimax(loadings, gamma=1.0, max_iter=100, tol=1e-6):
    """最大方差旋转（Kaiser 迭代）。"""
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


def factor_scores(X, n_factors=2):
    """主成分法提取因子 + varimax + 白化得分，返回 (载荷, 得分, 特征值)。"""
    R = np.corrcoef(X, rowvar=False)
    vals, vecs = np.linalg.eigh(R)
    o = np.argsort(vals)[::-1]
    vals, vecs = vals[o], vecs[:, o]
    L = vecs[:, :n_factors] * np.sqrt(vals[:n_factors])
    rot = varimax(L)
    for j in range(n_factors):
        if rot[:, j].sum() < 0:
            rot[:, j] *= -1
    return rot, X @ rot / np.sqrt(vals[:n_factors]), vals


def scatter_cities(x, y, labels, name):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=60, color="#8b7cff")
    for xx, yy, l in zip(x, y, labels):
        ax.text(xx, yy + .12, str(l), ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="gray", lw=.8); ax.axvline(0, color="gray", lw=.8)
    ax.set(xlabel="Gross", ylabel="Avg", title="10 城市因子得分")
    ax.grid(alpha=.3)
    fig.tight_layout()
    save(fig, name)


def main():
    # ---------- 1 贷款评分 PCA 打分 ----------
    loan = pd.read_csv(os.path.join(DATA, "Loan_aply.csv"))
    show("贷款评分数据", loan.shape)
    show("相关系数矩阵", loan.loc[:, "X1":].corr().round(3))
    Xl = StandardScaler().fit_transform(loan.loc[:, "X1":])
    pca_l = PCA(n_components=5).fit(Xl)
    show("5 个主成分解释方差比", pd.Series(pca_l.explained_variance_ratio_).round(4))
    show("教材手算校验：第一条样本在 PC1 上的得分",
         round(float(np.dot(Xl[0], pca_l.components_[0])), 4))
    pca1 = PCA(n_components=1).fit(Xl)
    score = loan[["ID"]].join(pd.DataFrame(pca1.transform(Xl), columns=["PCAScore"]))
    show("客户 PCA 综合得分", score.round(4))

    # ---------- 2 城市经济指标 ----------
    cities = pd.read_csv(os.path.join(DATA, "cities_10.csv"), encoding="gbk")
    feats = cities.loc[:, "X1":]
    X = StandardScaler().fit_transform(feats)
    show("城市指标相关系数（前 4 列）", feats.corr().iloc[:4, :4].round(3))

    pca = PCA(n_components=2).fit(X)
    show("PCA 累计解释方差比", np.cumsum(pca.explained_variance_ratio_).round(4))
    fig, ax = plt.subplots(figsize=(6, 4))
    all_pca = PCA().fit(X)
    ax.bar(range(1, len(all_pca.explained_variance_ratio_) + 1),
           all_pca.explained_variance_ratio_, color="#5aa9ff")
    ax.plot(range(1, len(all_pca.explained_variance_ratio_) + 1),
            np.cumsum(all_pca.explained_variance_ratio_), "-o", color="#ff7b7b")
    ax.set(title="城市数据碎石图", xlabel="主成分", ylabel="解释方差比")
    fig.tight_layout()
    save(fig, "PCA碎石图")

    # SparsePCA：不同 alpha 下非零载荷个数，教材用它选稀疏度
    rows = []
    for a in np.arange(.1, 1.0, .05):
        sp = SparsePCA(n_components=2, alpha=round(float(a), 2),
                       random_state=0).fit(X)
        nz = int(np.count_nonzero(sp.components_))
        rows.append({"alpha": round(float(a), 2), "非零载荷数": nz})
    show("SparsePCA 稀疏度选择", pd.DataFrame(rows).to_string(index=False))
    sp = SparsePCA(n_components=2, alpha=.65, random_state=0).fit(X)
    show("alpha=0.65 的稀疏载荷", pd.DataFrame(sp.components_.T,
         index=feats.columns, columns=["F1", "F2"]).round(3))

    # 因子分析
    rot, scores, vals = factor_scores(X, 2)
    show("旋转后因子载荷",
         pd.DataFrame(rot, index=feats.columns, columns=["Gross", "Avg"]).round(3))
    fa_df = pd.DataFrame(scores, columns=["Gross", "Avg"])
    result = cities.join(fa_df)
    show("因子得分（复现）", result[["AREA", "Gross", "Avg"]].round(3))
    ref = pd.read_csv(os.path.join(DATA, "cities_FA_score.csv"))
    show("与教材答案的相关系数", pd.Series({
        "Gross": np.corrcoef(result.Gross, ref.Gross)[0, 1],
        "Avg": np.corrcoef(result.Avg, ref.Avg)[0, 1]}).round(4))
    scatter_cities(result.Gross, result.Avg, result.AREA, "因子得分散点")

    fig, ax = plt.subplots(figsize=(8, 4))
    pd.DataFrame(rot, index=feats.columns, columns=["Gross", "Avg"]).plot(
        kind="bar", ax=ax, color=["#5aa9ff", "#ff7b7b"])
    ax.set(title="旋转后因子载荷", xlabel="指标", ylabel="载荷")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    save(fig, "因子载荷")

    # ---------- 3 变量聚类（等价替代 varSelec） ----------
    corr = feats.corr().abs().values
    D = 1 - corr
    np.fill_diagonal(D, 0)
    Z = sch.linkage(squareform(D, checks=False), method="average")
    labels = sch.fcluster(Z, t=2, criterion="maxclust")
    vc = pd.DataFrame({"变量": feats.columns, "簇": labels})
    # 每簇挑与其他变量平均相关最高的作为代表
    reps = []
    for g, sub in vc.groupby("簇"):
        cols = sub["变量"].tolist()
        rep = feats[cols].corr().abs().sum().idxmax()
        reps.append(rep)
        print(f"  簇 {g}: {cols}  -> 代表变量 {rep}")
    show("变量聚类选出的代表变量", reps)
    show("保留后的数据", feats[reps])

    fig, ax = plt.subplots(figsize=(8, 4))
    sch.dendrogram(Z, labels=list(feats.columns), ax=ax)
    ax.set(title="变量聚类树状图（按 1-|相关系数| 距离）", ylabel="距离")
    fig.tight_layout()
    save(fig, "变量聚类树状图")


if __name__ == "__main__" or True:
    main()
