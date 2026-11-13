# -*- coding: utf-8 -*-
"""
3.1（上）申请反欺诈：决策树 / 集成 / 经典分类 —— 合并教材两个主 notebook
======================================================================
loan_net.csv（社交网络特征预测欺诈 flag），教材用同一份数据比较了 10 种模型：
  决策树(GridSearch) → 随机森林 → AdaBoost → XGBoost → LightGBM
  → 标准化后的 KNN / 朴素贝叶斯 / SVM / MLP；
评估统一用 classification_report + ROC / PR / KS 曲线。

修正：pydotplus + graphviz → sklearn.tree.plot_tree / export_text；
教材里"寻优需把 sklearn 降到 1.6 以下"的警告在新版已不适用（xgboost 3.x 直接可用）；
网格搜索空间按原教材保留但缩小迭代规模，SVC(probability=True) 在 2.7 万行上很慢，
因此统一抽样 6000 行（SAMPLE_ROWS 可改 None 跑全量）。

数据：data/loan_net.csv。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, roc_curve, auc,
                             precision_recall_curve, average_precision_score,
                             roc_auc_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('loan_net.csv',)   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\敏捷数据挖掘复现\3.1申请反欺诈"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '3.1申请反欺诈'),
              os.path.join(os.getcwd(), '敏捷数据挖掘复现', '3.1申请反欺诈')]
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

SAMPLE_ROWS = 6000
RS = 1234


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def save(fig, name):
    p = os.path.join(CHARTS, f"3.1.1_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def metrics_curves(y_test, y_train, proba_test, proba_train, name):
    """教材的 ROC / PR / KS 三合一图。"""
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for yy, pp, lab, c in [(y_test, proba_test, "测试集", "#5aa9ff"),
                           (y_train, proba_train, "训练集", "#ff7b7b")]:
        fpr, tpr, _ = roc_curve(yy, pp)
        ax[0].plot(fpr, tpr, color=c, label=f"{lab} AUC={auc(fpr, tpr):.3f}")
        pr, rc, _ = precision_recall_curve(yy, pp)
        ax[1].plot(rc, pr, color=c, label=f"{lab} AP={average_precision_score(yy, pp):.3f}")
        ks = np.max(tpr - fpr)
        ax[2].plot(np.linspace(0, 1, len(tpr)), tpr, color=c, label=f"{lab} TPR")
        ax[2].plot(np.linspace(0, 1, len(fpr)), fpr, color=c, ls="--",
                   label=f"{lab} FPR (KS={ks:.3f})")
    ax[0].plot([0, 1], [0, 1], color="gray", lw=.8)
    for a, t in zip(ax, ["ROC", "Precision-Recall", "KS"]):
        a.set_title(t); a.set_xlabel("ratio"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.tight_layout()
    save(fig, name)


def main():
    raw = pd.read_csv(os.path.join(DATA, "loan_net.csv"), skipinitialspace=True)
    show("数据规模 / 欺诈比例", f"{raw.shape}，flag=1 占比 {raw.flag.mean():.2%}")
    df = raw.copy()
    if SAMPLE_ROWS:
        # 欺诈样本只有 71 条，随机抽样会几乎抽不到正例；保留全部正例再抽负例
        pos = df[df.flag == 1]
        neg = df[df.flag == 0].sample(n=min(SAMPLE_ROWS, (df.flag == 0).sum()),
                                      random_state=RS)
        df = pd.concat([pos, neg]).sample(frac=1, random_state=RS).reset_index(drop=True)
    X, y = df.iloc[:, :4], df.flag
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4,
                                          random_state=RS, stratify=y)
    show("训练 / 测试样本量", pd.Series({"train": len(Xtr), "test": len(Xte)}))

    results, probas = {}, {}

    def evaluate(name, model, Xtr_, Xte_):
        model.fit(Xtr_, ytr)
        pte = model.predict_proba(Xte_)[:, 1]
        ptr = model.predict_proba(Xtr_)[:, 1]
        results[name] = {"AUC": roc_auc_score(yte, pte),
                         "AP": average_precision_score(yte, pte),
                         "准确率": (model.predict(Xte_) == yte).mean()}
        probas[name] = (ptr, pte)
        return model

    # ---- 1 决策树 ----
    grid = {"criterion": ["gini", "entropy"], "max_depth": [2, 3, 4, 5],
            "min_samples_split": [2, 10], "min_samples_leaf": [2, 10]}
    dt_cv = GridSearchCV(DecisionTreeClassifier(random_state=RS), grid,
                         scoring="roc_auc", cv=4, n_jobs=-1)
    dt_cv.fit(Xtr, ytr)
    best_tree = dt_cv.best_estimator_
    show("决策树最优参数", dt_cv.best_params_)
    show("决策树分类报告", classification_report(yte, best_tree.predict(Xte), digits=3))
    evaluate("决策树", best_tree, Xtr, Xte)
    metrics_curves(yte, ytr, probas["决策树"][1], probas["决策树"][0], "决策树评估曲线")

    fig, ax = plt.subplots(figsize=(12, 5))
    plot_tree(best_tree, feature_names=list(X.columns), max_depth=3,
              filled=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title("最优决策树（展示 3 层）")
    fig.tight_layout()
    save(fig, "决策树结构")
    show("决策树规则（前 12 行）",
         "\n".join(export_text(best_tree, feature_names=list(X.columns),
                               max_depth=3).splitlines()[:12]))

    # ---- 2 集成方法 ----
    rf = RandomForestClassifier(n_estimators=60, max_depth=8,
                                min_samples_leaf=5, random_state=RS, n_jobs=-1)
    evaluate("随机森林", rf, Xtr, Xte)
    ada = AdaBoostClassifier(n_estimators=30, learning_rate=0.8, random_state=RS)
    evaluate("AdaBoost", ada, Xtr, Xte)
    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=.1,
                        eval_metric="auc", random_state=RS, n_jobs=-1,
                        verbosity=0)
    evaluate("XGBoost", xgb, Xtr, Xte)
    lgb = LGBMClassifier(n_estimators=100, num_leaves=15, learning_rate=.1,
                         random_state=RS, n_jobs=-1, verbose=-1)
    evaluate("LightGBM", lgb, Xtr, Xte)

    # ---- 3 经典分类（标准化后） ----
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    knn = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": [3, 7, 11, 15]},
                       scoring="roc_auc", cv=4, n_jobs=-1)
    knn.fit(Xtr_s, ytr)
    show("KNN 最优 k", knn.best_params_)
    evaluate("KNN", knn.best_estimator_, Xtr_s, Xte_s)
    evaluate("朴素贝叶斯", GaussianNB(), Xtr_s, Xte_s)
    svc = SVC(C=1.0, gamma="scale", kernel="rbf", probability=True, random_state=RS)
    evaluate("SVM", svc, Xtr_s, Xte_s)
    mlp = MLPClassifier(hidden_layer_sizes=(15,), activation="relu",
                        alpha=0.01, max_iter=400, random_state=RS)
    evaluate("MLP", mlp, Xtr_s, Xte_s)

    # ---- 4 汇总 ----
    summary = pd.DataFrame(results).T.sort_values("AUC", ascending=False).round(4)
    show("10 个模型的测试集表现", summary)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    summary.AUC.sort_values().plot(kind="barh", ax=ax, color="#5aa9ff")
    ax.set(title="各模型测试集 AUC", xlabel="AUC")
    fig.tight_layout()
    save(fig, "模型AUC对比")

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (_, pte) in probas.items():
        fpr, tpr, _ = roc_curve(yte, pte)
        ax.plot(fpr, tpr, lw=1.2, label=f"{name} ({auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=.8)
    ax.set(title="ROC 曲线汇总", xlabel="FPR", ylabel="TPR")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout()
    save(fig, "ROC汇总")

    out = os.path.join(CHARTS, "fraud_dtree.joblib")
    joblib.dump(best_tree, out)
    print(f"\n最优决策树已保存：{out}")


if __name__ == "__main__" or True:
    main()
