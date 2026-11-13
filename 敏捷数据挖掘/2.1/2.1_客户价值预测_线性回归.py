# -*- coding: utf-8 -*-
"""
2.1 客户价值预测（线性回归）—— 由《2.1线性回归.ipynb》整理成 .py
================================================================
汽车贷款/信用卡支出案例，教材完整走了一遍建模与诊断：
  简单线性回归 → 多元 → 前向逐步筛选 → 残差异方差 → 对数变换 → QQ 图
  → 强影响点剔除 → VIF 共线性 → 构造 high_avg_ratio → 加入离散变量
  → 样本外预测 → 岭回归 / Lasso / 弹性网络（含岭迹图和系数路径）。

数据：data/creditcard_exp.csv（上一期）、data/creditcard_exp_out_of_sample.csv（本期）。
修正：去掉 %matplotlib inline；图统一存 charts/；`oos_data['edu_class'].replace(..., inplace=True)`
在 pandas 3 里对链式结果无效，改成列赋值；对数变换前把非正值裁掉，避免 inf。

运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (RidgeCV, Ridge, LassoCV, Lasso, ElasticNetCV)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('creditcard_exp.csv', 'creditcard_exp_out_of_sample.csv')   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\敏捷数据挖掘复现\2.1客户价值预测"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '2.1客户价值预测'),
              os.path.join(os.getcwd(), '敏捷数据挖掘复现', '2.1客户价值预测')]
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
    p = os.path.join(CHARTS, f"2.1_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def forward_select(data, response):
    """按 AIC 前向选变量，每步加入让 AIC 下降最多的变量。"""
    remaining = sorted(c for c in data.columns if c != response)
    selected, current = [], float("inf")
    while remaining:
        scored = []
        for c in remaining:
            f = f"{response} ~ " + " + ".join(selected + [c])
            try:
                scored.append((ols(f, data).fit().aic, c))
            except Exception:
                continue
        if not scored:
            break
        best_aic, best = min(scored, key=lambda x: x[0])
        if best_aic >= current:
            break
        current, _ = best_aic, remaining.remove(best)
        selected.append(best)
        print(f"  + {best:<20s} AIC={best_aic:.2f}")
    return selected


def vif(df, y_col):
    cols = [c for c in df.columns if c != y_col]
    r2 = ols(f"{y_col} ~ " + " + ".join(cols), df).fit().rsquared
    return 1. / (1. - r2)


def plot_resid(resid, x, y_hat, name):
    """教材的残差-X / 残差-拟合值 双面板图。"""
    fig, ax = plt.subplots(2, 1, figsize=(7, 7))
    ax[0].scatter(x, resid, s=20, alpha=.6, color="#5aa9ff")
    ax[0].axhline(0, color="gray", lw=1)
    ax[0].set(title="残差 vs X", xlabel="X", ylabel="resid")
    ax[1].scatter(y_hat, resid, s=20, alpha=.6, color="#ff7b7b")
    ax[1].axhline(0, color="gray", lw=1)
    ax[1].set(title="残差 vs 拟合值", xlabel="y_hat", ylabel="resid")
    fig.tight_layout()
    save(fig, name)


def main():
    raw = pd.read_csv(os.path.join(DATA, "creditcard_exp.csv"), skipinitialspace=True)
    oos = pd.read_csv(os.path.join(DATA, "creditcard_exp_out_of_sample.csv"),
                      skipinitialspace=True)
    show("原始 / 样本外数据规模", f"{raw.shape} / {oos.shape}")
    show("训练数据字段", list(raw.columns))

    train = raw[raw.avg_exp.notnull()].copy().iloc[:, 2:].drop("age2", axis=1)
    oos = oos.drop("age2", axis=1)
    show("建模样本量", train.shape)
    show("连续变量相关系数", train[["Income", "avg_exp", "Age", "dist_home_val"]]
         .corr().round(3))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(train.Income, train.avg_exp, s=25, color="#5aa9ff")
    ax.set(title="收入与月均支出", xlabel="Income", ylabel="avg_exp")
    ax.grid(alpha=.3)
    save(fig, "收入与支出散点")

    # ---- 1 简单 / 多元线性回归 ----
    m1 = ols("avg_exp ~ Income", train).fit()
    show("简单线性回归参数", m1.params.round(4))
    show("简单线性回归 R² / AIC", pd.Series({"R2": m1.rsquared, "AIC": m1.aic}).round(3))
    show("样本外前 5 个预测", m1.predict(oos).head().round(2))

    m2 = ols("avg_exp ~ Income + dist_home_val + dist_avg_income", train).fit()
    show("多元线性回归参数", m2.params.round(4))
    show("多元线性回归 R²", round(m2.rsquared, 4))

    sel = forward_select(train[["avg_exp", "Income", "Age",
                                "dist_home_val", "dist_avg_income"]], "avg_exp")
    show("前向筛选保留的变量", sel)

    # ---- 2 异方差诊断：三种函数形式对比 ----
    ln = lambda s: np.log(s.clip(lower=1e-6))       # noqa: E731 避免 0/负值取 log
    train["Income_ln"] = ln(train.Income)
    oos["Income_ln"] = ln(oos.Income)
    lm1 = ols("avg_exp ~ Income", train).fit()
    lm2 = ols("avg_exp_ln ~ Income", train).fit()
    lm3 = ols("avg_exp_ln ~ Income_ln", train).fit()
    show("三种函数形式的 R²", pd.Series({
        "exp ~ Income": lm1.rsquared,
        "ln(exp) ~ Income": lm2.rsquared,
        "ln(exp) ~ ln(Income)": lm3.rsquared}).round(4))
    plot_resid(lm1.resid, train.Income, lm1.predict(train), "残差_原始模型")
    plot_resid(lm3.resid, train.Income_ln, lm3.predict(train), "残差_双对数模型")

    fig, ax = plt.subplots(figsize=(5, 5))
    sm.qqplot(lm3.resid, fit=True, line="45", ax=ax)
    ax.set_title("双对数模型残差 QQ 图")
    save(fig, "残差QQ图")

    # ---- 3 强影响点 ----
    sr = (lm3.resid - lm3.resid.mean()) / lm3.resid.std()
    show("|学生化残差| > 2 的样本数", int((sr.abs() > 2).sum()))
    train = train.loc[sr.abs() <= 2].copy()
    lm4 = ols("avg_exp_ln ~ Income_ln", train).fit()
    show("剔除强影响点后 R²", round(lm4.rsquared, 4))

    # ---- 4 共线性 ----
    for c in ("dist_home_val", "dist_avg_income"):
        train[c + "_ln"] = ln(train[c])
        oos[c + "_ln"] = ln(oos[c])
    exog = train[["Age", "Income_ln", "dist_home_val_ln", "dist_avg_income_ln"]]
    show("VIF（变换前）", pd.Series({c: vif(exog, c) for c in exog}).round(2))
    # 教材用"高出平均收入的比率"替换其中一个共线变量
    for d in (train, oos):
        d["high_avg_ratio"] = d["high_avg"] / d["dist_avg_income"]
    exog2 = train[["Age", "high_avg_ratio", "dist_home_val_ln", "dist_avg_income_ln"]]
    show("VIF（构造 high_avg_ratio 后）",
         pd.Series({c: vif(exog2, c) for c in exog2}).round(2))
    sel2 = forward_select(train[["avg_exp_ln", "Age", "high_avg_ratio",
                                 "dist_home_val_ln", "dist_avg_income_ln"]], "avg_exp_ln")
    show("第二轮前向筛选", sel2)

    # ---- 5 加入离散变量并预测样本外 ----
    f7 = ("avg_exp_ln ~ dist_avg_income_ln + dist_home_val_ln + "
          "C(gender) + C(Ownrent) + C(Selfempl) + C(edu_class)")
    m7 = ols(f7, train).fit()
    f8 = ("avg_exp_ln ~ dist_avg_income_ln + dist_home_val_ln + "
          "C(Selfempl) + C(gender) * C(edu_class)")
    m8 = ols(f8, train).fit()
    show("离散变量模型对比", pd.DataFrame({
        "R2": [m7.rsquared, m8.rsquared], "AIC": [m7.aic, m8.aic]},
        index=["主效应", "含交互项"]).round(3))
    oos["edu_class"] = oos["edu_class"].replace({0: 1})   # 本期 0 概化为 1
    oos["Pred_avg_exp"] = np.exp(m8.predict(oos))
    out = os.path.join(CHARTS, "样本外预测结果.csv")
    oos.to_csv(out, index=False, encoding="utf-8-sig")
    show("样本外预测前 5 条", oos[["id", "Pred_avg_exp"]].head().round(2))

    # ---- 6 正则算法 ----
    cont = ["Age", "Income", "dist_home_val", "dist_avg_income"]
    scaler = StandardScaler().fit(train[cont])
    X = scaler.transform(train[cont])
    X_oos = scaler.transform(oos[cont])
    y = train["avg_exp_ln"]
    alphas = np.logspace(-2, 3, 60)
    rcv = RidgeCV(alphas=alphas, cv=3).fit(X, y)
    coefs = np.array([Ridge(alpha=a).fit(X, y).coef_ for a in alphas])
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, c in enumerate(cont):
        ax.plot(alphas, coefs[:, j], label=c)
    ax.set_xscale("log")
    ax.set(title="岭迹图", xlabel="alpha", ylabel="系数")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    save(fig, "岭迹图")
    show("RidgeCV 最优 alpha / R²",
         pd.Series({"alpha": rcv.alpha_, "R2": rcv.score(X, y)}).round(4))

    l_alphas = np.logspace(-3, 0, 60)
    lcv = LassoCV(alphas=l_alphas, cv=5, max_iter=5000).fit(X, y)
    lcoefs = np.array([Lasso(alpha=a, max_iter=5000).fit(X, y).coef_
                       for a in l_alphas])
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, c in enumerate(cont):
        ax.plot(l_alphas, lcoefs[:, j], label=c)
    ax.set_xscale("log")
    ax.set(title="Lasso 系数路径", xlabel="alpha", ylabel="系数")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    save(fig, "Lasso系数路径")
    show("LassoCV 最优 alpha / 系数", pd.Series(lcv.coef_, index=cont).round(4))

    encv = ElasticNetCV(l1_ratio=[.1, .5, .9, .99, 1], cv=4,
                        max_iter=5000).fit(X, y)
    show("ElasticNetCV 结果", pd.Series({
        "l1_ratio": encv.l1_ratio_, "alpha": encv.alpha_,
        "R2": encv.score(X, y)}).round(4))
    show("样本外预测示例（Ridge）", np.exp(rcv.predict(X_oos)[:5]).round(2))
    print(f"\n样本外预测文件：{out}")


if __name__ == "__main__" or True:
    main()
