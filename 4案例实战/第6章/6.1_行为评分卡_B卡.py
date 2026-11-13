# -*- coding: utf-8 -*-
"""
第6章 贷中信用风险预测模型（B 卡）—— 由《第6章...代码通过.ipynb》整理
======================================================================
信用卡行为评分卡：从 12 期账单流水构造客户级特征，再用 toad 建评分卡。
  1) 读取 CreditBalance（账户×账期的余额/还款/消费/逾期）与 Label（账户额度与好坏标签）；
  2) 四类特征：还款表现 PaymentVar、逾期情况 DelqVar、额度使用 UseVar、
     账户信息 AccountInfo，按 Cust_ID 聚合；
  3) 回溯期 3/6/12 个月分别构造，拼成建模宽表 ABT + Target；
  4) 清洗(-9999 填补)、toad.detect、分层分区；
  5) 决策树分箱 + WOE + 相关性筛选 + 逐步回归 + 逻辑回归 + ROC/KS。

数据：教材 CreditBalance.csv 有 61.8 万行 / 28 MB，这里按账户抽 5000 个
Account_ID 的全部期数（65000 行）存成 CreditBalance_sample.csv，方法完全一致；
把文件名改回 CreditBalance.csv 并放全量数据即可跑全量。

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
_EXPECTED = ("CreditBalance_sample.csv", "label_sample.csv")


def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\金融数据分析案例实战复现\第6章-B卡评分卡"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(), os.path.join(os.getcwd(), "第6章-B卡评分卡"),
              os.path.join(os.getcwd(), "金融数据分析案例实战复现", "第6章-B卡评分卡")]
    for r in roots:
        d = os.path.join(r, "data")
        if os.path.isdir(d) and any(os.path.exists(os.path.join(d, x)) for x in _EXPECTED):
            return r
    return os.getcwd()


_HERE = _locate_chapter_dir()
DATA = os.path.join(_HERE, "data")
CHARTS = os.path.join(_HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)

BAL_FILE = "CreditBalance_sample.csv"
LAB_FILE = "label_sample.csv"
TOP_IV = 15
RS = 1234


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def save(fig, name):
    p = os.path.join(CHARTS, f"6.1_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def payment_var(bal):
    """还款表现类：还款率 = 还款额/账单余额（余额<=0 记缺失）的均值/极值/波动。"""
    d = bal.copy()
    d["Pay_Ratio"] = d.Payment / d.Balance.map(lambda x: np.nan if x <= 0 else x)
    g = d.groupby("Cust_ID").Pay_Ratio
    return pd.DataFrame({"Pay_Ratio_AVG": g.mean(), "Pay_Ratio_MAX": g.max(),
                         "Pay_Ratio_MIN": g.min(), "Pay_Ratio_STD": g.std()})


def delq_var(bal):
    """逾期类：各逾期级次数、占比、最大连续逾期。"""
    g = bal.groupby("Cust_ID")
    out = pd.DataFrame(index=g.size().index)
    for c in ["Delq1", "Delq2", "Delq3"]:
        out[f"{c}_Count"] = g[c].sum()
        out[f"{c}_Percent"] = g[c].mean()
    return out


def use_var(bal):
    """额度使用类：账单/消费占额度的均值与极值。"""
    d = bal.copy()
    d["Balance"] = d.Balance.map(lambda x: 0 if x < 0 else x)
    d["BillUseRate"] = d.Balance / d.Credit_Limit.replace(0, np.nan)
    d["SpendUseRate"] = d.Spend / d.Credit_Limit.replace(0, np.nan)
    g = d.groupby("Cust_ID")
    return pd.DataFrame({"BillUseRate_AVG": g.BillUseRate.mean(),
                         "BillUseRate_MAX": g.BillUseRate.max(),
                         "SpendUseRate_AVG": g.SpendUseRate.mean(),
                         "SpendUseRate_MAX": g.SpendUseRate.max()})


def account_info(lab):
    """账户信息类：账户数、额度合计/最大/最小，客户级标签（任一账户坏则坏）。"""
    g = lab.groupby("Cust_ID")
    return pd.DataFrame({"NumAcc": g.Account_ID.count(),
                         "CreditLimit_SUM": g.Credit_Limit.sum(),
                         "CreditLimit_MAX": g.Credit_Limit.max(),
                         "CreditLimit_MIN": g.Credit_Limit.min(),
                         "Target": g.label.max()})


def abt(bal, lab, windows=(3, 6, 12)):
    """按回溯期 3/6/12 个月构造客户级宽表（等价于教材 ABT 函数）。"""
    bal = bal.merge(lab[["Account_ID", "Credit_Limit"]], on="Account_ID", how="left")
    max_period = bal.Period.max()
    parts = []
    for T in windows:
        sub = bal[bal.Period > max_period - T]
        feats = pd.concat([payment_var(sub), delq_var(sub), use_var(sub)], axis=1)
        feats.columns = [f"{c}_{T}M" for c in feats.columns]
        parts.append(feats)
    wide = pd.concat(parts + [account_info(lab)], axis=1)
    return wide.reset_index().rename(columns={"index": "Cust_ID"})


def main():
    bal = pd.read_csv(os.path.join(DATA, BAL_FILE))
    lab = pd.read_csv(os.path.join(DATA, LAB_FILE))
    show("账单 / 标签规模", f"{bal.shape} / {lab.shape}")
    show("账期范围 / 客户数 / 坏客户占比",
         f"{bal.Period.min()}~{bal.Period.max()} / {lab.Cust_ID.nunique()} / "
         f"{lab.groupby('Cust_ID').label.max().mean():.2%}")

    dat = abt(bal, lab)
    show("ABT 宽表规模 / 特征数", f"{dat.shape} / {dat.shape[1]-2}")
    show("缺失率最高的 5 列", dat.isna().mean().sort_values(ascending=False).head(5).round(3))

    num = dat.drop(columns=["Cust_ID"]).apply(pd.to_numeric, errors="coerce")
    num = num.replace([np.inf, -np.inf], np.nan).fillna(-9999)
    num["Target"] = num["Target"].astype(int)
    show("数据质量报告（前 6 列）", toad.detect(num).T.head(6))

    X = num.drop(columns=["Target"])
    y = num["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.33, random_state=RS, stratify=y)
    show("训练 / 测试", f"{X_train.shape} / {X_test.shape}")

    # 特征初筛：去空、去常数列、按 IV 取前 TOP_IV 个
    t1 = toad.selection.drop_empty(X_train, threshold=.9)
    t2 = toad.selection.drop_var(t1, threshold=0)
    iv = toad.quality(t2, y_train, iv_only=True).sort_values("iv", ascending=False)
    show("IV 最高的 10 个特征", iv.head(10).round(4))
    keep = [c for c in iv.head(TOP_IV).index if c in t2.columns]
    X_train_1, X_test_1 = t2[keep].copy(), X_test[keep].copy()

    bins = toad.transform.Combiner()
    bins.fit(X_train_1, y_train, method="dt", n_bins=5, min_samples=.05)
    woe = toad.transform.WOETransformer()
    X_train_woe = woe.fit_transform(bins.transform(X_train_1), y_train).fillna(0)
    X_test_woe = woe.transform(bins.transform(X_test_1)).fillna(0)
    show("WOE 矩阵", f"{X_train_woe.shape} / {X_test_woe.shape}")

    try:
        X_train_woe_1 = toad.selection.drop_corr(X_train_woe, y_train, .6)
        X_test_woe_1 = X_test_woe[X_train_woe_1.columns]
        X_train_woe_2 = toad.selection.stepwise(X_train_woe_1, y_train,
                                                direction="both", criterion="aic")
        X_test_woe_2 = X_test_woe_1[X_train_woe_2.columns]
        show("逐步回归后入模特征", list(X_train_woe_2.columns))
    except Exception as e:
        print(f"  逐步回归跳过（{type(e).__name__}: {e}）")
        X_train_woe_2, X_test_woe_2 = X_train_woe, X_test_woe

    model = LogisticRegression(max_iter=1000).fit(X_train_woe_2, y_train)
    show("入模系数", pd.Series(model.coef_[0],
                             index=X_train_woe_2.columns).round(3))
    p_tr = model.predict_proba(X_train_woe_2)[:, 1]
    p_te = model.predict_proba(X_test_woe_2)[:, 1]
    fpr_tr, tpr_tr, _ = metrics.roc_curve(y_train, p_tr)
    fpr_te, tpr_te, _ = metrics.roc_curve(y_test, p_te)
    auc_tr, auc_te = metrics.auc(fpr_tr, tpr_tr), metrics.auc(fpr_te, tpr_te)
    ks = max(abs(tpr_te - fpr_te))
    show("模型评估", pd.Series({"train_AUC": auc_tr, "test_AUC": auc_te,
                              "test_KS": ks}).round(4))
    try:
        show("PSI（前 6）", toad.metrics.PSI(X_train_woe_2, X_test_woe_2).head(6).round(4))
    except Exception as e:
        print("  PSI 跳过：", type(e).__name__)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(fpr_tr, tpr_tr, label=f"train AUC={auc_tr:.3f}")
    ax[0].plot(fpr_te, tpr_te, label=f"test AUC={auc_te:.3f}")
    ax[0].plot([0, 1], [0, 1], "k--")
    ax[0].set(title="B 卡 ROC", xlabel="FPR", ylabel="TPR"); ax[0].legend(); ax[0].grid(alpha=.3)
    th = np.linspace(0, 1, len(tpr_te))
    ax[1].plot(th, tpr_te, label="TPR"); ax[1].plot(th, fpr_te, label="FPR")
    ax[1].plot(th, np.abs(tpr_te - fpr_te), label=f"KS={ks:.3f}")
    ax[1].set(title="B 卡 KS", xlabel="Population"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()
    save(fig, "ROC与KS曲线")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(p_te[y_test == 0], bins=30, alpha=.6, label="好客户", color="#3ddc97")
    ax.hist(p_te[y_test == 1], bins=30, alpha=.6, label="坏客户", color="#ff7b7b")
    ax.set(title="预测概率分布", xlabel="P(坏)", ylabel="客户数"); ax.legend()
    fig.tight_layout()
    save(fig, "预测概率分布")


if __name__ == "__main__" or True:
    main()
