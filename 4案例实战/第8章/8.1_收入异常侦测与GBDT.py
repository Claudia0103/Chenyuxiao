# -*- coding: utf-8 -*-
"""
第8章（上）申请反欺诈：收入异常侦测与 GBDT —— 合并主 notebook 前半与 test.ipynb
==============================================================================
思路：申请信息里收入、学历容易被篡改，用「收入对年龄/学历/城市/性别回归的残差」
衡量收入是否偏离同群体常态，残差绝对值就是异常特征，再喂给 GBDT 判欺诈。

流程：标准化 age / income 及其绝对值 → ols 回归取收入残差 → 标准化残差 + 绝对值
→ 按 flag 分组描述与箱线图 → 7:3 划分 → GBDT 网格搜索 → ROC / PR / KS 评估。

修正：
  - 教材 Mymetrics.py 里的 metrics_roc / metrics_pr 内联进本文件（不依赖外部模块）；
  - sklearn 1.9 里 GradientBoostingClassifier 的 `loss='deviance'` 已改名
    `'log_loss'`，且不再支持 `exponential`，网格搜索相应调整；
  - `test.ipynb` 读的是作者本机 C:/Users/wfyh/data12261.csv（数据不在教材里），
    它的 RFE + 逻辑回归流程并入这里作为对照模型。

数据：data/loan1_org_basic.csv。运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
CHAPTER_DIR = ""
_EXPECTED = ("loan1_org_basic.csv",)


def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\金融数据分析案例实战复现\第8章-申请反欺诈"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(), os.path.join(os.getcwd(), "第8章-申请反欺诈"),
              os.path.join(os.getcwd(), "金融数据分析案例实战复现", "第8章-申请反欺诈")]
    for r in roots:
        d = os.path.join(r, "data")
        if os.path.isdir(d) and any(os.path.exists(os.path.join(d, x)) for x in _EXPECTED):
            return r
    return os.getcwd()


_HERE = _locate_chapter_dir()
DATA = os.path.join(_HERE, "data")
CHARTS = os.path.join(_HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)

RS = 12345


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def save(fig, name):
    p = os.path.join(CHARTS, f"8.1_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def metrics_curves(y_test, y_train, proba_test, proba_train, name):
    """教材 Mymetrics.py 的 ROC / PR / KS 三合一图（内联等价实现）。"""
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for yy, pp, lab, c in [(y_test, proba_test, "测试集", "#5aa9ff"),
                           (y_train, proba_train, "训练集", "#ff7b7b")]:
        fpr, tpr, _ = metrics.roc_curve(yy, pp)
        ax[0].plot(fpr, tpr, color=c, label=f"{lab} AUC={metrics.auc(fpr, tpr):.3f}")
        pr, rc, _ = metrics.precision_recall_curve(yy, pp)
        ax[1].plot(rc, pr, color=c,
                   label=f"{lab} AP={metrics.average_precision_score(yy, pp):.3f}")
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
    data = pd.read_csv(os.path.join(DATA, "loan1_org_basic.csv"))
    show("数据规模 / 欺诈比例", f"{data.shape} / {data.flag.mean():.2%}")
    show("收入描述统计", data.income.describe().round(2))

    sc = StandardScaler()
    data["age_st"] = sc.fit_transform(data[["age"]])
    data["age_st_abs"] = np.abs(data.age_st)
    data["income_st"] = StandardScaler().fit_transform(data[["income"]])
    data["income_st_abs"] = np.abs(data.income_st)
    show("标准化收入按欺诈标签分组",
         data.income_st.groupby(data.flag).describe().round(3))

    # 收入残差：同年龄/学历/城市/性别群体里的"异常高或异常低"
    lm = smf.ols("income ~ age + C(edu) + C(city_level) + C(gender)", data).fit()
    show("收入回归 R²", round(lm.rsquared, 4))
    data["income_resid"] = lm.resid
    data["income__resid_st"] = StandardScaler().fit_transform(data[["income_resid"]])
    data["income__resid_st_abs"] = np.abs(data["income__resid_st"])
    show("残差绝对值按标签分组均值",
         data.groupby("flag").income__resid_st_abs.mean().round(3))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(x="flag", y="income_resid", data=data, ax=ax[0])
    ax[0].set_title("收入残差 vs 欺诈标签")
    sns.boxplot(x="flag", y="income__resid_st_abs", data=data, ax=ax[1])
    ax[1].set_title("标准化残差绝对值 vs 欺诈标签")
    fig.tight_layout()
    save(fig, "收入异常特征")

    feats = ["age_st", "age_st_abs", "income_st", "income_st_abs",
             "income_resid", "income__resid_st", "income__resid_st_abs",
             "gender", "city_level", "edu"]
    feats = [c for c in feats if c in data.columns]
    X, y = data[feats], data.flag
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3, train_size=.7,
                                          random_state=RS, stratify=y)
    show("训练 / 测试", f"{Xtr.shape} / {Xte.shape}")

    param_grid = {"learning_rate": [0.1, 0.2], "n_estimators": [80],
                  "max_depth": [2, 3]}
    gbc = GridSearchCV(GradientBoostingClassifier(random_state=RS),
                       param_grid, scoring="roc_auc", cv=4, n_jobs=-1)
    gbc.fit(Xtr, ytr)
    show("GBDT 最优参数", gbc.best_params_)
    p_tr = gbc.predict_proba(Xtr)[:, 1]
    p_te = gbc.predict_proba(Xte)[:, 1]
    show("GBDT AUC / KS", pd.Series({
        "train_AUC": metrics.roc_auc_score(ytr, p_tr),
        "test_AUC": metrics.roc_auc_score(yte, p_te),
        "test_KS": max(np.abs(metrics.roc_curve(yte, p_te)[1]
                              - metrics.roc_curve(yte, p_te)[0]))}).round(4))
    metrics_curves(yte, ytr, p_te, p_tr, "GBDT评估曲线")

    # test.ipynb 的 RFE + 逻辑回归对照
    rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=5)
    rfe.fit(Xtr, ytr)
    lr = LogisticRegression(max_iter=1000).fit(Xtr.iloc[:, rfe.get_support()], ytr)
    p_lr = lr.predict_proba(Xte.iloc[:, rfe.get_support()])[:, 1]
    show("RFE 选中的特征", list(Xtr.columns[rfe.get_support()]))
    show("RFE + 逻辑回归 test AUC",
         round(metrics.roc_auc_score(yte, p_lr), 4))
    show("分类报告（GBDT）", metrics.classification_report(
        yte, gbc.predict(Xte), digits=3))


if __name__ == "__main__" or True:
    main()
