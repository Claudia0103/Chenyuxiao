# -*- coding: utf-8 -*-
"""
2.2 用户响应预测（逻辑回归）—— 由《2.2逻辑回归.ipynb》整理成 .py
================================================================
开卡响应预测，教材流程：statsmodels 一元 logit 与多阈值评估 → WOE/IV 变量筛选
→ sklearn 逻辑回归 → RFECV 递归特征消除 → LogisticRegressionCV 调参
→ 混淆矩阵 / 分类报告 / ROC / AUC → 样本外预测 → joblib 模型持久化。

关于 WOE：教材明确写了「使用 woe 包会把 pandas 降到 1.5，本方法不建议使用」，
所以这里不用教材附带的 woe.py，改成自己写的等频分箱 WOE/IV 计算（结果等价、无降版本风险）。

数据：data/creditcard_exp.csv、data/creditcard_exp_out_of_sample.csv。
修正：去掉 %matplotlib inline；`replace(..., inplace=True)` 改列赋值；阈值循环里
pd.crosstab 先 reindex 补齐行列，避免极端阈值下 .loc[0,0] KeyError。

运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import (roc_curve, auc, classification_report,
                             confusion_matrix)
import joblib

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
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\敏捷数据挖掘复现\2.2用户响应预测"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '2.2用户响应预测'),
              os.path.join(os.getcwd(), '敏捷数据挖掘复现', '2.2用户响应预测')]
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

CONT = ["Age", "Income", "dist_home_val", "dist_avg_income", "high_avg"]
DISC = ["gender", "Ownrent", "Selfempl", "edu_class"]


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def save(fig, name):
    p = os.path.join(CHARTS, f"2.2_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def woe_iv(x, y, bins=5):
    """等频分箱计算 WOE 与 IV：WOE=ln(好占比/坏占比)，IV=Σ(好占比-坏占比)*WOE。"""
    try:
        b = pd.qcut(x, bins, duplicates="drop")
    except ValueError:
        return np.nan
    tab = pd.crosstab(b, y).reindex(columns=[0, 1], fill_value=0).astype(float)
    good = (tab[0] + .5) / (tab[0].sum() + .5 * len(tab))
    bad = (tab[1] + .5) / (tab[1].sum() + .5 * len(tab))
    w = np.log(good / bad)
    return float(((good - bad) * w).sum())


def threshold_table(y_true, proba):
    rows = []
    for t in np.arange(0.1, 1.0, 0.1):
        pred = (proba > t).astype(int)
        cm = pd.crosstab(pred, y_true, margins=True).reindex(
            index=[0, 1, "All"], columns=[0, 1, "All"], fill_value=0)
        precision = cm.loc[0, 0] / cm.loc["All", 0]
        recall = cm.loc[0, 0] / cm.loc[0, "All"]
        spec = cm.loc[1, 1] / cm.loc[1, "All"]
        f1 = 2 * precision * recall / (precision + recall)
        rows.append({"阈值": round(float(t), 2), "precision": round(precision, 3),
                     "recall": round(recall, 3), "Specificity": round(spec, 3),
                     "f1": round(f1, 3)})
    return pd.DataFrame(rows)


def main():
    data = pd.read_csv(os.path.join(DATA, "creditcard_exp.csv"), skipinitialspace=True)
    oos = pd.read_csv(os.path.join(DATA, "creditcard_exp_out_of_sample.csv"),
                      skipinitialspace=True)
    data = data.drop(["avg_exp", "avg_exp_ln", "age2"], axis=1)
    oos = oos.drop("age2", axis=1)
    oos["edu_class"] = oos["edu_class"].replace({0: 1})
    show("数据规模 / 开卡比例", f"{data.shape}，Acc=1 占比 {data.Acc.mean():.2%}")

    train = data.sample(frac=.7, random_state=1234).copy()
    test = data[~data.index.isin(train.index)].copy()
    show("训练集 / 测试集", pd.Series({"train": len(train), "test": len(test)}))

    # ---- 1 statsmodels 一元逻辑回归 ----
    lr = smf.logit("Acc ~ Income", train).fit(disp=0)
    show("一元 logit 参数", lr.params.round(4))
    test["Pred_Acc"] = lr.predict(test)
    show("多阈值指标（一元模型）", threshold_table(test.Acc, test.Pred_Acc))

    # ---- 2 WOE / IV 变量筛选 ----
    iv = pd.Series({c: woe_iv(train[c], train.Acc)
                    for c in CONT + DISC if c in train.columns}).dropna()
    show("各变量 IV（教材用 woe 包算，这里自实现等频分箱）", iv.round(4))
    fig, ax = plt.subplots(figsize=(7, 4))
    iv.sort_values().plot(kind="barh", ax=ax, color="#5aa9ff")
    ax.set(title="变量 IV 值", xlabel="Information Value")
    fig.tight_layout()
    save(fig, "变量IV值")

    # ---- 3 sklearn 逻辑回归 + RFECV ----
    cols = CONT + DISC
    scaler = StandardScaler().fit(train[cols])
    for d in (train, test, oos):
        d[cols] = scaler.transform(d[cols])
    base = LogisticRegression(max_iter=1000)
    base.fit(train[cols], train.Acc)
    show("全变量模型测试集预测（前 10）", base.predict(test[cols])[:10])

    rfecv = RFECV(base, cv=3, scoring="accuracy", n_jobs=-1)
    rfecv.fit(train[cols], train.Acc)
    selected = list(np.array(cols)[rfecv.support_])
    show("RFECV 特征排名", dict(zip(cols, rfecv.ranking_)))
    show("RFECV 保留变量", selected)

    # ---- 4 交叉验证调参 ----
    lrcv = LogisticRegressionCV(Cs=10, cv=4, n_jobs=-1, max_iter=1000)
    lrcv.fit(train[selected], train.Acc)
    show("LogisticRegressionCV 最优 C", lrcv.C_)
    test_proba = lrcv.predict_proba(test[selected])[:, 1]
    train_proba = lrcv.predict_proba(train[selected])[:, 1]
    pred = lrcv.predict(test[selected])
    show("分类报告", classification_report(test.Acc, pred, digits=3))
    show("混淆矩阵", pd.DataFrame(confusion_matrix(test.Acc, pred),
                               index=["实际0", "实际1"], columns=["预测0", "预测1"]))

    fpr, tpr, _ = roc_curve(test.Acc, test_proba)
    fpr_tr, tpr_tr, _ = roc_curve(train.Acc, train_proba)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, "b-", label=f"测试集 AUC={auc(fpr, tpr):.4f}")
    ax.plot(fpr_tr, tpr_tr, "r--", label="训练集")
    ax.plot([0, 1], [0, 1], color="gray", lw=.8)
    ax.set(title="ROC 曲线", xlabel="FPR", ylabel="TPR")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    save(fig, "ROC曲线")

    # ---- 5 样本外预测与模型持久化 ----
    oos["Pred_Acc"] = lrcv.predict_proba(oos[selected])[:, 1]
    out_csv = os.path.join(CHARTS, "样本外预测结果.csv")
    oos.to_csv(out_csv, index=False, encoding="utf-8-sig")
    show("样本外预测前 5 条", oos[["id", "Pred_Acc"]].head().round(4))

    model_path = os.path.join(CHARTS, "logistic_model.joblib")
    joblib.dump(lrcv, model_path)
    reloaded = joblib.load(model_path)
    show("模型保存并重新加载后的预测一致",
         bool(np.allclose(reloaded.predict_proba(oos[selected])[:, 1],
                          oos["Pred_Acc"].values)))
    print(f"\n输出：{out_csv}、{model_path}")


if __name__ == "__main__" or True:
    main()
