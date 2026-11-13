# -*- coding: utf-8 -*-
"""
7 类别不平衡问题 —— 由《类别不平衡专题_课堂代码.ipynb》整理成 .py
==================================================================
  上半：用 make_blobs 造 1000:100 的不平衡数据，逐一演示
        SMOTE 过采样、RandomUnderSampler、TomekLinks、ENN、
        SMOTETomek / SMOTEENN 联合采样，画出各方法的数据分布；
  下半：真实银行营销数据集 train_set.csv，比较
        不采样 / SMOTE / SMOTENC / XGBoost 的 scale_pos_weight /
        max_delta_step 五种处理方式的 classification_report 与 AUC。

修正：imblearn 0.14 的 SMOTENC 需要类别特征的**位置索引**，
教材传的是列名列表（`list(cat_cols.columns)`）会报错 → 改用索引；
`pd.DataFrame(Y).value_counts()` 结果打印更清楚地改成 pd.Series；
xgb.train 用 sklearn 接口 XGBClassifier 包装，避免 DMatrix 与 pandas 3 的类型摩擦。

数据：data/train_set.csv（18987 行）。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import (RandomUnderSampler, TomekLinks,
                                     EditedNearestNeighbours)
from imblearn.combine import SMOTETomek, SMOTEENN
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('train_set.csv',)   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\敏捷数据挖掘复现\7类别不平衡"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '7类别不平衡'),
              os.path.join(os.getcwd(), '敏捷数据挖掘复现', '7类别不平衡')]
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

RS = 42


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def save(fig, name):
    p = os.path.join(CHARTS, f"7_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def demo_sampling():
    X, y = make_blobs(n_samples=[1000, 100], centers=[[0, 0], [2.0, 2.0]],
                      cluster_std=[1.5, 1.0], random_state=RS)
    show("原始类别分布", pd.Series(y).value_counts().sort_index())

    methods = [
        ("SMOTE 过采样", SMOTE(random_state=RS, sampling_strategy=.5)),
        ("Random 欠采样", RandomUnderSampler(random_state=RS, sampling_strategy=.3)),
        ("TomekLinks", TomekLinks()),
        ("ENN 编辑近邻", EditedNearestNeighbours(n_neighbors=20)),
        ("SMOTE + Tomek", SMOTETomek(random_state=RS)),
        ("SMOTE + ENN", SMOTEENN(random_state=RS)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (name, sampler) in zip(axes.ravel(), methods):
        Xs, ys = sampler.fit_resample(X, y)
        ax.scatter(Xs[:, 0], Xs[:, 1], c=ys, cmap="rainbow", s=6, alpha=.6)
        ax.set_title(f"{name}\n{pd.Series(ys).value_counts().to_dict()}", fontsize=10)
        ax.grid(alpha=.2)
        show(f"{name} 采样后类别分布", pd.Series(ys).value_counts().sort_index())
    fig.suptitle("六种采样方法对比（教材散点图）", fontsize=13)
    fig.tight_layout()
    save(fig, "六种采样方法对比")


def real_case():
    raw = pd.read_csv(os.path.join(DATA, "train_set.csv"))
    show("原始数据", f"{raw.shape}，y=1 占比 {raw.y.mean():.2%}")
    df = raw.drop(columns=["ID"]).dropna().reset_index(drop=True)
    labels = df.pop("y").astype(int)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_idx = [df.columns.get_loc(c) for c in cat_cols]
    if cat_cols:
        df[cat_cols] = OrdinalEncoder(handle_unknown="use_encoded_value",
                                      unknown_value=-1).fit_transform(
                                          df[cat_cols].astype(str))
    Xtr, Xte, ytr, yte = train_test_split(df, labels, test_size=.3,
                                          stratify=labels, random_state=RS)
    show("训练 / 测试正类占比", f"{ytr.mean():.3%} / {yte.mean():.3%}")
    show("类别特征索引（SMOTENC 用）", cat_idx)

    def fit_eval(name, X_fit, y_fit, params=None):
        params = params or {}
        m = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=.1,
                          eval_metric="auc", random_state=RS, n_jobs=-1,
                          **params)
        m.fit(X_fit, y_fit)
        proba = m.predict_proba(Xte)[:, 1]
        pred = (proba > .5).astype(int)
        rep = classification_report(yte, pred, output_dict=True, zero_division=0)
        return {"方案": name, "AUC": roc_auc_score(yte, proba),
                "正类precision": rep["1"]["precision"],
                "正类recall": rep["1"]["recall"],
                "正类f1": rep["1"]["f1-score"]}

    rows = [fit_eval("不采样基线", Xtr, ytr)]
    X_sm, y_sm = SMOTE(random_state=RS).fit_resample(Xtr, ytr)
    show("SMOTE 后训练集分布", pd.Series(y_sm).value_counts().sort_index())
    rows.append(fit_eval("SMOTE 过采样", X_sm, y_sm))

    try:
        smnc = SMOTENC(categorical_features=cat_idx, random_state=RS)
        X_sn, y_sn = smnc.fit_resample(Xtr, ytr)
        rows.append(fit_eval("SMOTENC（类别列专用）", X_sn, y_sn))
    except Exception as e:
        print(f"  SMOTENC 跳过：{type(e).__name__}: {e}")

    weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    show("scale_pos_weight（负/正样本比）", round(weight, 2))
    rows.append(fit_eval("scale_pos_weight", Xtr, ytr,
                         {"scale_pos_weight": weight}))
    rows.append(fit_eval("max_delta_step", Xtr, ytr,
                         {"max_delta_step": 0.6}))

    summary = pd.DataFrame(rows).set_index("方案").round(4)
    show("五种处理方式的测试集表现", summary)
    fig, ax = plt.subplots(figsize=(9, 4))
    summary[["AUC", "正类f1", "正类recall"]].plot(kind="bar", ax=ax,
                                                   color=["#5aa9ff", "#8b7cff", "#3ddc97"])
    ax.set(title="不平衡处理方案对比", ylabel="指标值", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig, "不平衡方案对比")
    print("\n教材结论：关心排序/AUC 可用过采样或 scale_pos_weight；"
          "关心预测概率校准时，重采样会改变标签分布，不建议使用。")


def main():
    print("7 类别不平衡：采样方法 → 真实数据上的代价敏感对比")
    demo_sampling()
    real_case()


if __name__ == "__main__" or True:
    main()
