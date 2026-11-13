# -*- coding: utf-8 -*-
"""
第2章 某银行贷款产品精准营销模型 —— 合并教材 2 个 notebook
==========================================================
教材前半用 MySQL 把 KDD99 的 8 张表拼成建模宽表（loan_y / test2 / test_pred），
后半用 Python 做清洗、盖帽法、逻辑回归和样本外预测。

复现做法：
  - SQL 建宽表用 pandas 等价重写（读 mysql_csv/ 下 6 张小表，GBK、无表头，
    列名按 create_table.sql 补上）；57 MB 的交易明细 Kdd99_trans.csv 没有复制，
    交易类特征直接用教材已产出的 tran0917.csv；
  - SQL 原文保存在 data/ 下（1 create_table.sql、第二章用到的sql.txt），供对照；
  - test.ipynb 是作者读本机 C:/Users/wfyh/data12261.csv 的草稿，数据不在教材里，
    它的 RFE + 逻辑回归流程并入本脚本的建模部分。

数据：data/tran0917.csv、pred0917.csv、mysql_csv/*.csv。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import sklearn.metrics as metrics

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
CHAPTER_DIR = ""       # 需要手动指定时填章节目录绝对路径
_EXPECTED = ("tran0917.csv", "pred0917.csv", "mysql_csv")


def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\金融数据分析案例实战复现\第2章-精准营销"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(), os.path.join(os.getcwd(), "第2章-精准营销"),
              os.path.join(os.getcwd(), "金融数据分析案例实战复现", "第2章-精准营销")]
    for r in roots:
        d = os.path.join(r, "data")
        if os.path.isdir(d) and any(os.path.exists(os.path.join(d, x)) for x in _EXPECTED):
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


def rebuild_wide_table():
    """SQL 的 loan_y / test2 用 pandas 等价重建：账户 → 贷款标签 + 静态特征。"""
    csv = os.path.join(DATA, "mysql_csv")
    spec = {
        "kdd99_accounts.csv": ["account_id", "district_id", "frequency", "date"],
        "Kdd99_clients.csv": ["client_id", "sex", "birth_date", "district_id"],
        "Kdd99_disp.csv": ["disp_id", "client_id", "account_id", "type"],
        "Kdd99_district.csv": ["A1", "GDP", "A4", "A10", "A11", "A12",
                               "A13", "A14", "A15", "a16"],
        "Kdd99_card.csv": ["card_id", "disp_id", "issued", "type"],
        "Kdd99_loans.csv": ["loan_id", "account_id", "date", "amount",
                            "duration", "payments", "status"],
    }
    tabs = {}
    for f, cols in spec.items():
        p = os.path.join(csv, f)
        if os.path.exists(p):
            tabs[f.split(".")[0].lower()] = pd.read_csv(
                p, encoding="gbk", header=None, names=cols)
    show("读入的 KDD99 表", {k: v.shape for k, v in tabs.items()})
    acc = tabs["kdd99_accounts"].copy()
    loans = tabs["kdd99_loans"].rename(columns={"date": "loan_date"})
    # y: 0=无贷款, 2=1998 前已有贷款, 1=1998 后新贷款（等同教材 SQL 的 case when）
    merged = acc.merge(loans, on="account_id", how="left")
    merged["y"] = np.where(merged.loan_date.isna(), 0,
                           np.where(merged.loan_date < "1998-01-01", 2, 1))
    disp = tabs["kdd99_disp"].query("type == '所有者'")
    clients = tabs["kdd99_clients"]
    district = tabs["kdd99_district"]
    card = tabs["kdd99_card"]
    wide = (merged.merge(disp[["disp_id", "client_id", "account_id"]],
                         on="account_id", how="left")
            .merge(clients[["client_id", "sex", "birth_date"]],
                   on="client_id", how="left")
            .merge(district[["A1", "GDP", "A4"]],
                   left_on="district_id", right_on="A1", how="left")
            .merge(card[["disp_id", "issued"]], on="disp_id", how="left"))
    wide["age"] = 1998 - pd.to_datetime(wide.birth_date, errors="coerce").dt.year
    wide["avg_GDP"] = wide.GDP / wide.A4
    wide["card"] = (pd.to_datetime(wide.issued, errors="coerce")
                    < pd.Timestamp("1998-01-01")).astype(int)
    show("重建宽表字段", list(wide.columns))
    show("y 分布（0=无贷款 1=1998后 2=1998前）", wide.y.value_counts().sort_index())
    out = os.path.join(CHARTS, "SQL等价宽表.csv")
    wide.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  宽表已保存：{os.path.basename(out)}")
    return wide


def cap_series(s, low=0.01, high=0.99):
    """教材盖帽法：小于 1% 分位、大于 99% 分位的值拉回分位数。"""
    q1, q99 = s.quantile([low, high])
    return s.clip(q1, q99)


def main():
    rebuild_wide_table()

    snd = pd.read_csv(os.path.join(DATA, "tran0917.csv"))
    show("建模宽表规模", snd.shape)
    show("字段", list(snd.columns))
    show("描述统计（数值列）", snd.describe().T.round(2).head(10))
    if snd.y.nunique() > 2:      # SQL 里 y=2 是观察期前已有贷款，不进二分类模型
        snd = snd[snd.y != 2].copy()
        show("剔除 y=2 后样本量 / y 分布", f"{len(snd)} / {snd.y.value_counts().to_dict()}")

    # 缺失即"没发生该业务"，填 0（教材口径）
    num_cols = ["avg_balance", "std_balance", "cv_balance",
                "sum_expenditure", "sum_income", "in_out_ratio"]
    for c in num_cols:
        if c in snd:
            snd[c] = snd[c].fillna(0)
    show("填补后缺失合计", int(snd.isna().sum().sum()))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(x="y", y="cv_balance", data=snd, ax=ax[0])
    ax[0].set_title("按标签看 cv_balance")
    ax[1].hist(snd.std_balance, bins=40, color="#5aa9ff")
    ax[1].set_title("盖帽前 std_balance")
    fig.tight_layout()
    save(fig, "数据探查")

    before = snd.std_balance.copy()
    snd["std_balance"] = cap_series(snd.std_balance)
    snd["cv_balance"] = cap_series(snd.cv_balance)
    show("std_balance 盖帽前后极值",
         pd.DataFrame({"前": [before.min(), before.max()],
                       "后": [snd.std_balance.min(), snd.std_balance.max()]},
                      index=["min", "max"]).round(3))

    train = snd.sample(frac=.7, random_state=1234).copy()
    test = snd[~snd.index.isin(train.index)].copy()
    show("训练 / 测试样本量", pd.Series({"train": len(train), "test": len(test)}))

    formula = "y ~ avg_balance + cv_balance + sum_expenditure + C(card)"
    lg = smf.logit(formula, train).fit(disp=0)
    show("逻辑回归参数", lg.params.round(4))
    show("逻辑回归 AIC / 伪 R²",
         pd.Series({"AIC": lg.aic, "prsquared": lg.prsquared}).round(4))

    train["proba"] = lg.predict(train)
    test["proba"] = lg.predict(test)
    fpr, tpr, _ = metrics.roc_curve(test.y, test.proba)
    fpr_tr, tpr_tr, _ = metrics.roc_curve(train.y, train.proba)
    auc_test, auc_train = metrics.auc(fpr, tpr), metrics.auc(fpr_tr, tpr_tr)
    show("AUC", pd.Series({"训练集": auc_train, "测试集": auc_test}).round(4))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, "b--", label=f"test AUC={auc_test:.4f}")
    ax.plot(fpr_tr, tpr_tr, "r-", label=f"train AUC={auc_train:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=.8)
    ax.set(title="ROC 曲线", xlabel="FPR", ylabel="TPR")
    ax.legend(); ax.grid(alpha=.3)
    save(fig, "ROC曲线")

    # 样本外：对 pred0917.csv 打分（教材 C42）
    pred = pd.read_csv(os.path.join(DATA, "pred0917.csv"))
    for c in ["avg_balance", "cv_balance", "sum_expenditure"]:
        if c in pred:
            pred[c] = pred[c].fillna(0)
    if "cv_balance" in pred:
        pred["cv_balance"] = cap_series(pred.cv_balance)
    pred["Pred_y"] = lg.predict(pred)
    show("样本外预测前 5 条", pred[["account_id", "Pred_y"]].head().round(4))
    out = os.path.join(CHARTS, "样本外预测.csv")
    pred.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  预测结果：{os.path.basename(out)}")


if __name__ == "__main__" or True:
    main()
