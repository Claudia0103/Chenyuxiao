# -*- coding: utf-8 -*-
"""
6 特征工程 —— 由《Feature-engineering.ipynb》整理成 .py
========================================================
电信账单数据，教材按"预处理 → 特征构造 → 特征提取 → 特征选择"走了一整遍：
  错误值 / 异常值(3σ 盖帽) / 对数变换 / 缺失填补 / 二值化 / 等宽等频人工离散 /
  哑变量 / 标准化归一化 / 构造单价与 TAC / PCA 与 LDA / Filter-Wrapper-Embedded 选择。

修正：pandas 3 里 `fillna(..., inplace=True)` 改列赋值；`FunctionTransformer` 输出
在新版 sklearn 是 ndarray，直接用 np.log1p 得到同样结果；get_dummies 的 bool 列
统一 astype(int)；groupby 分箱加 observed=False。

数据：data/telecom_bill.csv（3000 行）。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import (Binarizer, OneHotEncoder, StandardScaler,
                                   MinMaxScaler, Normalizer)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import (VarianceThreshold, chi2, SelectKBest,
                                       f_classif, f_regression, RFE,
                                       SelectFromModel)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('telecom_bill.csv',)   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\敏捷数据挖掘复现\6特征工程"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '6特征工程'),
              os.path.join(os.getcwd(), '敏捷数据挖掘复现', '6特征工程')]
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
    p = os.path.join(CHARTS, f"6_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def main():
    raw = pd.read_csv(os.path.join(DATA, "telecom_bill.csv"))
    show("原始数据", raw.shape)
    show("字段与类型", raw.dtypes.astype(str))
    show("错误值：traffic >= 5000 的条数", int((raw.traffic >= 5000).sum()))
    data = raw[raw.traffic < 5000].copy()
    show("state 分布", data.state.value_counts(dropna=False))
    fig, ax = plt.subplots(figsize=(5, 5))
    data.state.value_counts().plot(kind="pie", autopct="%.1f%%", ax=ax)
    ax.set(title="state 分布", ylabel="")
    save(fig, "state分布")
    data["state"] = data["state"].replace([2, 4], np.nan)

    # ---- 异常值：3σ 盖帽 ----
    mu, sigma = data.expenditure.mean(), data.expenditure.std()
    lb, hb = mu - 3 * sigma, mu + 3 * sigma
    capped = data.expenditure.clip(lb, hb)
    show("3σ 盖帽边界 / 被处理条数",
         f"[{lb:.1f}, {hb:.1f}] / {int(((data.expenditure < lb) | (data.expenditure > hb)).sum())}")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].boxplot(data.expenditure.dropna()); ax[0].set_title("原始 expenditure")
    ax[1].boxplot(capped.dropna()); ax[1].set_title("盖帽后")
    save(fig, "异常值盖帽前后")

    # ---- 对数变换 ----
    data["log_expd"] = np.log1p(capped)
    data["log_call"] = np.log1p(data["call"])
    data["log_traffic"] = np.log1p(data["traffic"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].hist(data.expenditure.dropna(), bins=30, color="#5aa9ff")
    ax[0].set_title("原始支出分布")
    ax[1].hist(data.log_expd.dropna(), bins=30, color="#3ddc97")
    ax[1].set_title("log1p 变换后")
    fig.tight_layout()
    save(fig, "对数变换前后")

    # ---- 缺失填补 ----
    show("缺失统计", data.isna().sum()[lambda s: s > 0])
    top_state = data.state.mode()[0]
    data = data.fillna({"IMEI": 0, "state": top_state})
    show("填补后缺失合计", int(data.isna().sum().sum()))

    # ---- 二值化 / 离散化 ----
    data["has_IMEI"] = Binarizer(threshold=0).fit_transform(data[["IMEI"]]).astype(int)
    show("has_IMEI 分布", data.has_IMEI.value_counts())
    duration = (datetime(2015, 1, 1) - pd.to_datetime(data.join_time)).map(
        lambda x: x.days / 30.)
    data["duration"] = duration
    equal_width = pd.cut(data.duration, 5)
    eq_freq = pd.qcut(data.duration, 8, duplicates="drop")
    artificial = pd.cut(data.duration, [-np.inf, 5, 12, 36, np.inf])
    show("等宽离散各箱 churn 率",
         data.churn.groupby(equal_width, observed=False).mean().round(3))
    show("等频离散各箱 churn 率",
         data.churn.groupby(eq_freq, observed=False).mean().round(3))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    data.churn.groupby(artificial, observed=False).mean().astype(float).plot(
        kind="bar", ax=ax, color="#8b7cff")
    ax.set(title="人工分箱（在网时长）的 churn 率", ylabel="churn 率")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save(fig, "人工分箱churn率")

    # ---- 哑变量与标准化 ----
    dummies = pd.get_dummies(data.state, prefix="s").astype(int)
    show("state 哑变量列", list(dummies.columns))
    show("OneHotEncoder 前 3 行",
         OneHotEncoder(sparse_output=False).fit_transform(data[["state"]])[:3])
    cols = ["log_expd", "log_call", "log_traffic"]
    std = StandardScaler().fit_transform(data[cols].fillna(0))
    mm = MinMaxScaler().fit_transform(data[cols].fillna(0))
    norm = Normalizer().fit_transform(std)
    show("StandardScaler / MinMax / Normalizer 第一行",
         pd.DataFrame({"std": std[0], "minmax": mm[0], "normalize": norm[0]},
                      index=cols).round(3))
    data1 = data.join(pd.DataFrame(std, columns=["s_expd", "s_call", "s_traffic"],
                                   index=data.index))

    # ---- 特征构造与提取 ----
    data1["unit_price"] = data1.expenditure / (data1.traffic + 1.)
    data1["TAC"] = data1.IMEI.map(lambda x: str(x)[:6])
    show("构造特征示例", data1[["expenditure", "traffic", "unit_price", "TAC"]].head())
    feats = ["s_expd", "s_call", "s_traffic"]
    pca = PCA(n_components=2, whiten=True).fit(data1[feats])
    show("PCA 解释方差比 / 成分", f"{pca.explained_variance_ratio_.round(3)} / "
                                  f"{pca.components_.round(3).tolist()}")
    lda = LinearDiscriminantAnalysis().fit(data1[feats], data1.churn)
    show("LDA 投影方向 / 作为分类器的准确率",
         f"{lda.coef_.round(3).tolist()} / "
         f"{(lda.predict(data1[feats]) == data1.churn).mean():.4f}")

    # ---- 特征选择 ----
    xc = data1[["vip", "has_IMEI"]].fillna(0)
    xn = data1[["s_expd", "s_call", "s_traffic", "duration"]].fillna(0)
    ycat, ycont = data1.churn, data1.s_expd
    vt = VarianceThreshold(threshold=.09).fit(xc)
    show("方差过滤（各变量方差 / 保留）",
         f"{vt.variances_.round(3)} / {vt.get_support().tolist()}")
    show("卡方检验（离散特征 vs churn）",
         pd.DataFrame({"chi2": chi2(xc.abs(), ycat)[0],
                       "p": chi2(xc.abs(), ycat)[1]}, index=xc.columns).round(4))
    show("ANOVA f_classif（连续特征 vs churn）",
         pd.Series(f_classif(xn, ycat)[0], index=xn.columns).round(2))
    show("f_regression（连续特征 vs 消费金额）",
         pd.Series(f_regression(xn, ycont)[0], index=xn.columns).round(2))

    sel_chi = SelectKBest(chi2, k=1).fit(xc.abs(), ycat)
    sel_f = SelectKBest(f_classif, k=2).fit(xn, ycat)
    rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=2).fit(xn, ycat)
    rfc = RandomForestClassifier(n_estimators=30, random_state=0, n_jobs=-1).fit(xn, ycat)
    emb = SelectFromModel(GradientBoostingClassifier(random_state=0),
                          threshold="mean").fit(xn, ycat)
    show("Filter 选中的离散特征（卡方 k=1）",
         pd.Series(sel_chi.get_support(), index=xc.columns))
    show("Wrapper / Embedded 选中的连续特征", pd.DataFrame({
        "ANOVA(k=2)": sel_f.get_support(),
        "RFE": rfe.get_support(),
        "随机森林重要性": rfc.feature_importances_.round(3),
        "GBDT SelectFromModel": emb.get_support(),
    }, index=xn.columns))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    pd.Series(rfc.feature_importances_, index=xn.columns).sort_values().plot(
        kind="barh", ax=ax, color="#5aa9ff")
    ax.set(title="随机森林特征重要性", xlabel="importance")
    fig.tight_layout()
    save(fig, "特征重要性")


if __name__ == "__main__" or True:
    main()
