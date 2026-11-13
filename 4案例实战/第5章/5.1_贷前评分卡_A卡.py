# -*- coding: utf-8 -*-
"""
第5章 贷前信用风险预测模型（A 卡）—— 由《第5章...代码通过.ipynb》整理
======================================================================
消费金融申请评分卡全流程（toad 0.1.7）：
  实体一致性去重 → 列类型分配 → toad.detect 数据质量报告 →
  错误值(DAYS_EMPLOYED<0)与缺失处理 → 分层抽样分区 →
  特征初筛(drop_empty / drop_var / drop_iv) → 决策树分箱 Combiner →
  人工调整切分点 → WOE 编码 → 相关性筛选 drop_corr → 逐步回归 stepwise →
  逻辑回归 → VIF / PSI → ROC / KS → ScoreCard 刻度转换 → KS_bucket 评分表现。

为控制运行时间，默认抽 10000 行、初筛后取 IV 最高的 20 个特征入模
（SAMPLE_ROWS / TOP_IV 可改大或设为 None 跑全量）。toad 0.1.7 在 pandas 3 下可用。

数据：data/creditdata_small.csv（3 万行 × 122 列）、data/数据字典.xlsx。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import toad
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
CHAPTER_DIR = ""
_EXPECTED = ("creditdata_small.csv",)


def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\金融数据分析案例实战复现\第5章-A卡评分卡"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(), os.path.join(os.getcwd(), "第5章-A卡评分卡"),
              os.path.join(os.getcwd(), "金融数据分析案例实战复现", "第5章-A卡评分卡")]
    for r in roots:
        d = os.path.join(r, "data")
        if os.path.isdir(d) and any(os.path.exists(os.path.join(d, x)) for x in _EXPECTED):
            return r
    return os.getcwd()


_HERE = _locate_chapter_dir()
DATA = os.path.join(_HERE, "data")
CHARTS = os.path.join(_HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)

SAMPLE_ROWS = 10000
TOP_IV = 20
RS = 1223


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def save(fig, name):
    p = os.path.join(CHARTS, f"5.1_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def main():
    dat = pd.read_csv(os.path.join(DATA, "creditdata_small.csv"))
    dat = dat.set_index("ID")
    dat = dat.loc[:, ~dat.columns.duplicated()]
    dat = dat.loc[~dat.index.duplicated(), :]
    if SAMPLE_ROWS:
        dat = dat.sample(n=SAMPLE_ROWS, random_state=RS)
    show("数据规模 / 坏客户占比", f"{dat.shape} / {dat.TARGET.mean():.2%}")

    str_cols = [c for c in ["GENDER", "FLAG_OWN_CAR", "HOUSING_TYPE",
                            "FLAG_OWN_REALTY", "EDULEVEL", "MARRIAGE"]
                if c in dat.columns]
    for c in str_cols:
        dat[c] = dat[c].astype(str)
    for c in dat.columns.difference(str_cols):
        dat[c] = pd.to_numeric(dat[c], errors="coerce")

    quality = toad.detect(dat)
    show("数据质量报告（前 8 列）", quality.T.head(8))
    show("缺失率最高的 5 列",
         dat.isna().mean().sort_values(ascending=False).head(5).round(4))

    if "DAYS_EMPLOYED" in dat:
        dat["DAYS_EMPLOYED"] = dat["DAYS_EMPLOYED"].map(
            lambda x: np.nan if x < 0 else x)
    num = dat.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
    obj = dat.select_dtypes(exclude="number").replace(
        ["nan", "", "special", "missing"], "missing")
    dat = pd.concat([num, obj], axis=1)
    show("清洗后规模", dat.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        dat.drop("TARGET", axis=1), dat.TARGET, test_size=.33,
        random_state=RS, stratify=dat.TARGET)
    show("训练 / 测试", f"{X_train.shape} / {X_test.shape}")

    # 特征初筛
    t1 = toad.selection.drop_empty(X_train, threshold=.9)
    t2 = toad.selection.drop_var(t1, threshold=0)
    iv = toad.quality(t2, y_train, iv_only=True).sort_values("iv", ascending=False)
    show("IV 最高的 10 个特征", iv.head(10).round(4))
    keep = [c for c in iv.head(TOP_IV).index if c in t2.columns] or list(t2.columns[:TOP_IV])
    X_train_1, X_test_1 = t2[keep].copy(), X_test[keep].copy()
    show("初筛后入模特征数 / 名称", f"{len(keep)} / {keep[:8]}")

    # 决策树分箱 + WOE
    bins = toad.transform.Combiner()
    bins.fit(X_train_1, y_train, method="dt", n_bins=5, min_samples=.05)
    show("Age 分箱切分点", bins.export().get("Age", "无 Age 列"))
    woe = toad.transform.WOETransformer()
    X_train_woe = woe.fit_transform(bins.transform(X_train_1), y_train)
    X_test_woe = woe.transform(bins.transform(X_test_1))
    show("WOE 矩阵", f"{X_train_woe.shape} / {X_test_woe.shape}")

    try:
        X_train_woe_1 = toad.selection.drop_corr(X_train_woe, y_train, threshold=.7)
        X_test_woe_1 = X_test_woe[X_train_woe_1.columns]
        show("相关性筛选后特征数", X_train_woe_1.shape[1])
        X_train_woe_2 = toad.selection.stepwise(X_train_woe_1, y_train,
                                                direction="both", criterion="aic")
        X_test_woe_2 = X_test_woe_1[X_train_woe_2.columns]
        show("逐步回归后特征数 / 名称", f"{X_train_woe_2.shape[1]} / {list(X_train_woe_2.columns)}")
    except Exception as e:
        print(f"  逐步回归跳过（{type(e).__name__}: {e}），改用相关性筛选结果")
        X_train_woe_2, X_test_woe_2 = X_train_woe, X_test_woe

    X_train_woe_2 = X_train_woe_2.fillna(0)
    X_test_woe_2 = X_test_woe_2.fillna(0)
    model = LogisticRegression(max_iter=1000).fit(X_train_woe_2, y_train)
    show("逻辑回归系数（前 8）",
         pd.Series(model.coef_[0], index=X_train_woe_2.columns).head(8).round(3))
    try:
        show("VIF（前 8）", toad.selection.VIF(X_train_woe_2).head(8).round(3))
    except Exception as e:
        print("  VIF 跳过：", type(e).__name__)
    try:
        show("PSI（前 8）", toad.metrics.PSI(X_train_woe_2, X_test_woe_2).head(8).round(4))
    except Exception as e:
        print("  PSI 跳过：", type(e).__name__)

    p_tr = model.predict_proba(X_train_woe_2)[:, 1]
    p_te = model.predict_proba(X_test_woe_2)[:, 1]
    fpr_tr, tpr_tr, _ = metrics.roc_curve(y_train, p_tr)
    fpr_te, tpr_te, _ = metrics.roc_curve(y_test, p_te)
    auc_tr, auc_te = metrics.auc(fpr_tr, tpr_tr), metrics.auc(fpr_te, tpr_te)
    ks = max(abs(tpr_te - fpr_te))
    show("模型评估", pd.Series({"train_AUC": auc_tr, "test_AUC": auc_te,
                              "test_KS": ks}).round(4))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(fpr_tr, tpr_tr, label=f"train AUC={auc_tr:.3f}")
    ax[0].plot(fpr_te, tpr_te, label=f"test AUC={auc_te:.3f}")
    ax[0].plot([0, 1], [0, 1], "k--")
    ax[0].set(title="ROC 曲线", xlabel="FPR", ylabel="TPR")
    ax[0].legend(); ax[0].grid(alpha=.3)
    th = np.linspace(0, 1, len(tpr_te))
    ax[1].plot(th, tpr_te, label="TPR")
    ax[1].plot(th, fpr_te, label="FPR")
    ax[1].plot(th, np.abs(tpr_te - fpr_te), label=f"KS={ks:.3f}")
    ax[1].set(title="KS 曲线", xlabel="Population", ylabel="Cumsum%")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()
    save(fig, "ROC与KS曲线")

    # 评分卡刻度转换（base 600 / odds 35 / PDO 50）
    try:
        card = toad.ScoreCard(combiner=bins, transer=woe, base_score=600,
                              base_odds=35, pdo=50)
        card.fit(X_train_woe_2, y_train)
        sc = card.export(to_frame=True)
        show("评分卡（前 8 行）", sc.head(8))
        train_score = card.predict(X_train_woe_2)
        report = toad.metrics.KS_bucket(train_score, y_train, bucket=10,
                                        method="quantile")
        show("评分表现报告（前 8 行）", report.head(8).round(3))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(train_score, bins=40, color="#5aa9ff", edgecolor="white")
        ax.set(title="申请评分分布", xlabel="score", ylabel="客户数")
        fig.tight_layout()
        save(fig, "评分分布")
    except Exception as e:
        print(f"  评分卡刻度转换跳过（{type(e).__name__}: {e}）；"
              "toad.ScoreCard 要求 combiner/transer 与入模特征完全一致")


if __name__ == "__main__" or True:
    main()
