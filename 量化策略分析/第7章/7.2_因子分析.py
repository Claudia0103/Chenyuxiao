# -*- coding: utf-8 -*-
"""
第7章 因子分析 —— 由《在 Python 中实现因子分析.ipynb》整理成 .py
================================================================
场景：10 个城市、9 个经济指标，先用 PCA 判断保留几个成分，再用因子分析
把指标压成"总量 Gross"和"人均 Avg"两个因子，最后画城市散点。

关于 fa_kit：教材用第三方包 fa_kit（FactorAnalysis / plotting），这个包不在
教材目录里、PyPI 上也装不到。这里按教材教的同一套算法自己实现，流程一一对应：
  load_data_samples + extract_components  -> 相关矩阵特征分解，取前 n 个成分
  find_comps_to_retain(method='top_n')    -> 保留前 2 个
  rotate_components(method='varimax')     -> 最大方差旋转（Kaiser 迭代）
  comps['rot']                            -> 旋转后的载荷矩阵
  np.dot(标准化数据, 旋转载荷)             -> 因子得分（教材原文的计算方式）
载荷的符号方向本身不唯一，这里固定成"载荷总和为正"，保证每次运行同一结果。

数据：data/cities_10.csv（GBK 编码，10 城市 × 9 指标）。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\量化策略分析复现\第7章-特征降维方法"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '第7章-特征降维方法'),
              os.path.join(os.getcwd(), '量化策略分析复现', '第7章-特征降维方法')]
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
    """最大方差旋转：找正交矩阵 T，让载荷矩阵每列的平方载荷方差最大。"""
    p, k = loadings.shape
    rotation = np.eye(k)
    old = 0.0
    for _ in range(max_iter):
        rotated = loadings @ rotation
        # Kaiser 的经典迭代公式
        u, s, vh = np.linalg.svd(
            loadings.T @ (rotated ** 3 - (gamma / p) * rotated @
                          np.diag(np.diag(rotated.T @ rotated))))
        rotation = u @ vh
        if old != 0 and s.sum() / old < 1 + tol:
            break
        old = s.sum()
    return loadings @ rotation, rotation


def factor_scores(X, n_factors=2):
    """主成分法提取因子 + varimax 旋转 + 得分，返回 (旋转载荷, 得分, 特征值)。

    X 是已经中心标准化的样本×变量矩阵。
    """
    R = np.corrcoef(X, rowvar=False)
    vals, vecs = np.linalg.eigh(R)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    raw = vecs[:, :n_factors] * np.sqrt(vals[:n_factors])
    rot, _ = varimax(raw)
    # 固定符号，避免特征向量正负翻转导致结果不稳定
    for j in range(n_factors):
        if rot[:, j].sum() < 0:
            rot[:, j] *= -1
    scores = X @ rot
    return raw, rot, scores, vals


def main():
    model_data = pd.read_csv(os.path.join(DATA, "cities_10.csv"), encoding="gbk")
    show("10 城市 × 9 指标", model_data)
    features = model_data.loc[:, "X1":]
    show("相关系数矩阵（判断是否有降维必要）", features.corr().round(3))

    X = StandardScaler().fit_transform(features)
    pca = PCA(n_components=9).fit(X)
    show("9 个主成分解释的方差（标准化后 >1 才值得保留）",
         pd.Series(pca.explained_variance_, index=[f"PC{i+1}" for i in range(9)]).round(3))
    show("解释方差比与累计", pd.DataFrame({
        "解释方差比": pca.explained_variance_ratio_.round(4),
        "累计": np.cumsum(pca.explained_variance_ratio_).round(4)},
        index=[f"PC{i+1}" for i in range(9)]))

    n_factors = 2
    raw, rot, scores, vals = factor_scores(X, n_factors)
    names = [f"{c} {LABELS.get(c, c)}" for c in features.columns]
    show("旋转前载荷（主成分法）",
         pd.DataFrame(raw, index=names, columns=["F1", "F2"]).round(3))
    loadings = pd.DataFrame(rot, index=names, columns=["Gross", "Avg"])
    show("varimax 旋转后载荷（对应 fa.comps['rot']）", loadings.round(3))

    fa_score = pd.DataFrame(scores, columns=["Gross", "Avg"])
    cities = model_data.join(fa_score)
    show("因子得分拼回原表（前 10 行）", cities[["AREA", "Gross", "Avg"]].round(3))

    # 教材的载荷折线图：每个因子在 9 个指标上的权重
    fig, ax = plt.subplots(figsize=(9, 4))
    for j, col in enumerate(["Gross", "Avg"]):
        ax.plot(range(len(names)), rot[:, j], "-o", label=col,
                color=["#5aa9ff", "#ff7b7b"][j])
    ax.axhline(0, color="gray", lw=.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([c.split()[0] for c in names])
    ax.set(title="旋转后因子载荷（X1~X9）", xlabel="指标", ylabel="载荷")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    p1 = os.path.join(CHARTS, "7.2_旋转后因子载荷.png")
    fig.savefig(p1, dpi=120); plt.close(fig)

    # 教材最后那张城市散点图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(cities.Gross, cities.Avg, s=60, color="#8b7cff")
    for x, y, label in zip(cities.Gross, cities.Avg, cities.AREA):
        ax.text(x, y + .12, str(label), ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="gray", lw=.8); ax.axvline(0, color="gray", lw=.8)
    ax.set(title="10 城市因子得分散点（Gross × Avg）",
           xlabel="Gross 总量因子", ylabel="Avg 人均因子")
    ax.grid(alpha=.3)
    fig.tight_layout()
    p2 = os.path.join(CHARTS, "7.2_城市因子得分散点.png")
    fig.savefig(p2, dpi=120); plt.close(fig)
    print(f"\n图已保存：{os.path.basename(p1)}、{os.path.basename(p2)}")


if __name__ == "__main__" or True:
    main()
