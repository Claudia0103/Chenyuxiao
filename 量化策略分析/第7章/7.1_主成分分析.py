# -*- coding: utf-8 -*-
"""
第7章 主成分分析 —— 由《在 Python 中实现主成分分析.ipynb》整理成 .py
====================================================================
场景：电信运营商想按客户在电话、短信、微信、上网四类业务上的使用量做分群，
先看看这 4 个变量能不能压成 2 个主成分。

教材流程：相关矩阵 → scale 标准化 → PCA(n_components=2, whiten=True) →
查看解释方差比、成分矩阵、主成分得分。这里补上碎石图和得分散点图，
把"压成了什么"画出来。

数据：data/profile_telecom.csv（600 个客户，4 个业务量字段）。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('profile_telecom.csv',)   # 该脚本会用到的 data/ 下的文件或子目录

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


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def main():
    profile = pd.read_csv(os.path.join(DATA, "profile_telecom.csv"))
    show("数据规模", profile.shape)
    show("前 5 行", profile.head())
    data = profile.loc[:, "cnt_call":]
    show("4 个业务量的相关系数矩阵", data.corr(method="pearson").round(4))

    X = scale(data)
    pca_all = PCA().fit(X)
    show("全部主成分解释方差比",
         pd.Series(pca_all.explained_variance_ratio_,
                   index=[f"PC{i+1}" for i in range(X.shape[1])]).round(4))
    show("累计解释方差比", np.cumsum(pca_all.explained_variance_ratio_).round(4))

    pca = PCA(n_components=2, whiten=True).fit(X)
    show("教材结果：前 2 个主成分解释方差比", pca.explained_variance_ratio_.round(4))
    show("教材结果：成分矩阵（行=主成分，列=原变量）",
         pd.DataFrame(pca.components_, columns=data.columns,
                      index=["PC1", "PC2"]).round(4))
    scores = pca.transform(X)
    show("教材结果：前 5 个客户的主成分得分",
         pd.DataFrame(scores[:5], columns=["PC1", "PC2"]).round(4))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(range(1, len(pca_all.explained_variance_ratio_) + 1),
              pca_all.explained_variance_ratio_, color="#5aa9ff")
    ax[0].plot(range(1, len(pca_all.explained_variance_ratio_) + 1),
               np.cumsum(pca_all.explained_variance_ratio_),
               "-o", color="#ff7b7b")
    ax[0].set(title="碎石图（柱=单个，线=累计）", xlabel="主成分", ylabel="解释方差比")
    ax[0].set_xticks(range(1, len(pca_all.explained_variance_ratio_) + 1))
    ax[1].scatter(scores[:, 0], scores[:, 1], s=12, alpha=.5, color="#8b7cff")
    ax[1].set(title="客户主成分得分散点", xlabel="PC1", ylabel="PC2")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    out = os.path.join(CHARTS, "7.1_主成分碎石图与得分.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\n图已保存：{out}")


if __name__ == "__main__" or True:
    main()
