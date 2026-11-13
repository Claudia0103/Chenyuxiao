# -*- coding: utf-8 -*-
"""
第8章 逻辑回归 —— 由《逻辑回归.ipynb》整理成 .py
================================================
电信客户流失预测，教材分两段：
  一元逻辑回归：交叉表 / 卡方 → 7:3 划分 → logit(churn~duration) →
                阈值、混淆矩阵、准确率、多阈值指标、ROC/AUC、约登指数最优阈值；
  多元逻辑回归：连续变量前向逐步筛选(AIC) + 分类变量卡方筛选 →
                最终模型 → 评估 → VIF 共线性检查。

对教材代码的修正：
  1. `percConvert(ser)` 里的 `ser[-1]` 在 pandas 3 是按标签取值会 KeyError → .iloc[-1]；
  2. 教材阈值循环直接 `cm.loc[0,0]`，某些阈值下 crosstab 缺 0/1 行会 KeyError
     → 先 reindex 补齐行列；
  3. forward_select 用 set 遍历，顺序不确定 → 改成排序后遍历；单个候选不收敛时跳过；
  4. 教材用 `get_ipython()` 之类 notebook 专属写法 → 去掉；
  5. `.fit()` 加 disp=0、maxiter，避免刷屏和迭代不足。

数据：data/telecom_churn.csv（3463 条）。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn import metrics

# 前向筛选里个别候选模型最大似然不收敛，属正常现象（该候选不会被选入），
# 只屏蔽这一类警告，避免刷屏；模型本身的收敛情况仍可通过 AIC 判断。
warnings.filterwarnings("ignore", category=ConvergenceWarning)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('telecom_churn.csv',)   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\量化策略分析复现\第8章-统计检验与预测"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '第8章-统计检验与预测'),
              os.path.join(os.getcwd(), '量化策略分析复现', '第8章-统计检验与预测')]
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


def fit_logit(formula, data):
    """教材的 smf.logit(...).fit()，统一不打印迭代过程、放宽迭代次数。"""
    return smf.logit(formula=formula, data=data).fit(disp=0, maxiter=100)


def threshold_table(y_true, proba, thresholds=None):
    """教材那张多阈值指标表，补齐缺失行列防止 KeyError。"""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.1)
    rows = []
    for t in thresholds:
        pred = (proba > t).astype(int)
        cm = pd.crosstab(pred, y_true, margins=True).reindex(
            index=[0, 1, "All"], columns=[0.0, 1.0, "All"], fill_value=0)
        precision = cm.loc[0, 0.0] / cm.loc["All", 0.0]
        recall = cm.loc[0, 0.0] / cm.loc[0, "All"]
        specificity = cm.loc[1, 1.0] / cm.loc[1, "All"]
        f1 = 2 * precision * recall / (precision + recall)
        rows.append({"阈值": round(float(t), 2), "precision": round(precision, 3),
                     "recall": round(recall, 3), "Specificity": round(specificity, 3),
                     "f1": round(f1, 3)})
    return pd.DataFrame(rows)


def forward_select(data, response):
    """按 AIC 做前向逐步筛选：每次加入让 AIC 下降最多的变量。"""
    remaining = sorted(c for c in data.columns if c != response)
    selected, current = [], float("inf")
    while remaining:
        scored = []
        for cand in remaining:
            f = f"{response} ~ " + " + ".join(selected + [cand])
            try:
                scored.append((fit_logit(f, data).aic, cand))
            except Exception:
                continue
        if not scored:
            break
        best_aic, best = min(scored, key=lambda x: x[0])
        if best_aic < current:
            remaining.remove(best)
            selected.append(best)
            current = best_aic
            print(f"  + {best:<14s} AIC={best_aic:.2f}")
        else:
            print("  AIC 不再下降，停止筛选")
            break
    formula = f"{response} ~ " + " + ".join(selected)
    print(f"  最终公式：{formula}")
    return fit_logit(formula, data)


def vif(df, col_i):
    cols = [c for c in df.columns if c != col_i]
    r2 = smf.ols(f"{col_i} ~ " + " + ".join(cols), df).fit().rsquared
    return 1. / (1. - r2)


def roc_auc(ax, train, test, title):
    fpr_t, tpr_t, th_t = metrics.roc_curve(test.churn, test.proba)
    fpr_r, tpr_r, _ = metrics.roc_curve(train.churn, train.proba)
    ax.plot(fpr_t, tpr_t, "b--", label=f"测试集 AUC={metrics.auc(fpr_t, tpr_t):.4f}")
    ax.plot(fpr_r, tpr_r, "r-", label="训练集")
    ax.plot([0, 1], [0, 1], color="gray", lw=.8)
    ax.set(title=title, xlabel="FPR", ylabel="TPR")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    return th_t[(tpr_t - fpr_t).argmax()]


def main():
    churn = pd.read_csv(os.path.join(DATA, "telecom_churn.csv"))
    show("数据规模", churn.shape)
    show("流失分布", churn.churn.value_counts())

    # ---- 分类变量相关关系 ----
    cross = pd.crosstab(churn.posTrend, churn.churn, margins=True)
    show("posTrend × churn 交叉表", cross)
    perc = lambda ser: ser / ser.iloc[-1]      # 教材 ser[-1] 的现代写法
    show("行百分比", cross.apply(perc, axis=1).round(4))
    chi2, p, dof, expected = stats.chi2_contingency(cross.iloc[:2, :2])
    print(f"\n卡方检验：chisq={chi2:.4f}, p-value={p:.4f}, dof={dof}")
    print(f"p {'<' if p < .05 else '>='} 0.05，"
          f"{'拒绝独立假设' if p < .05 else '不能拒绝独立假设'}")

    # ---- 划分数据集 ----
    train = churn.sample(frac=.7, random_state=666).copy()
    test = churn[~churn.index.isin(train.index)].copy()
    show("训练集 / 测试集样本量", pd.Series({"训练集": len(train), "测试集": len(test)}))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(churn.duration, churn.churn, s=8, alpha=.3, color="#5aa9ff")
    ax.set(title="在网时长与是否流失", xlabel="duration", ylabel="churn")
    fig.tight_layout()
    ax.figure.savefig(os.path.join(CHARTS, "8.4_在网时长与流失散点.png"), dpi=120)
    plt.close(fig)

    # ---- 一元逻辑回归 ----
    lg = fit_logit("churn ~ duration", train)
    show("一元逻辑回归 summary", lg.summary())
    train["proba"] = lg.predict(train)
    test["proba"] = lg.predict(test)
    show("测试集预测概率示例", test[["churn", "proba"]].head())

    test["prediction"] = (test.proba > .3).astype(int)
    show("阈值 0.3 的混淆矩阵", pd.crosstab(test.churn, test.prediction, margins=True))
    acc = (test.prediction == test.churn).mean()
    print(f"\n一元模型准确率：{acc:.4f}")
    show("多阈值指标（教材循环）", threshold_table(test.churn, test.proba))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    best1 = roc_auc(axes[0], train, test, "一元逻辑回归 ROC")
    show("一元模型约登指数最优阈值", round(float(best1), 4))

    # ---- 多元逻辑回归 ----
    candidates = ["churn", "duration", "AGE", "incomeCode", "nrProm",
                  "peakMinAv", "peakMinDiff", "call_10086"]
    show("连续变量前向筛选过程", "见下方逐行输出")
    lg_m1 = forward_select(train[candidates], response="churn")
    show("前向筛选模型参数", lg_m1.params.round(4))
    show(f"原来 {len(candidates)-1} 个变量，筛选后 {len(lg_m1.params)-1} 个自变量",
         list(lg_m1.params.index[1:]))

    class_col = ["posTrend", "prom", "edu_class", "feton", "curPlan", "avgplan",
                 "gender", "negTrend", "planChange", "posPlanChange", "negPlanChange"]
    pvals = {}
    for c in class_col:
        try:
            tab = pd.crosstab(churn[c], churn.churn)
            pvals[c] = stats.chi2_contingency(tab)[1]
        except Exception as e:
            pvals[c] = np.nan
            print(f"  {c} 卡方检验跳过：{type(e).__name__}")
    show("分类变量卡方检验 p 值", pd.Series(pvals).round(4))
    show("p<0.05 的分类变量", [k for k, v in pvals.items() if v < .05])

    formula = ("churn ~ duration + AGE + incomeCode + peakMinDiff + call_10086 "
               "+ C(edu_class) + C(feton) + C(posTrend) + C(negTrend) "
               "+ C(curPlan) + C(avgplan) + C(gender)")
    lg_m = fit_logit(formula, train)
    show("多元逻辑回归 summary", lg_m.summary())
    train["proba"] = lg_m.predict(train)
    test["proba"] = lg_m.predict(test)
    test["prediction"] = (test.proba > .5).astype(int)
    show("阈值 0.5 的混淆矩阵", pd.crosstab(test.churn, test.prediction, margins=True))
    print(f"\n多元模型准确率：{(test.prediction == test.churn).mean():.4f}")
    show("多阈值指标（多元模型）", threshold_table(test.churn, test.proba))

    best2 = roc_auc(axes[1], train, test, "多元逻辑回归 ROC")
    show("多元模型约登指数最优阈值", round(float(best2), 4))
    test["prediction"] = (test.proba > best2).astype(int)
    show("最优阈值下的混淆矩阵", pd.crosstab(test.churn, test.prediction, margins=True))
    fig.tight_layout()
    p1 = os.path.join(CHARTS, "8.4_一元与多元ROC对比.png")
    fig.savefig(p1, dpi=120); plt.close(fig)

    show("连续变量 VIF", pd.Series({
        c: vif(train[candidates].drop(columns=["churn"]), c)
        for c in candidates if c != "churn"}).round(3))
    print(f"\n图已保存：{os.path.basename(p1)}")


if __name__ == "__main__" or True:
    main()
