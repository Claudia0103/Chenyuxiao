# -*- coding: utf-8 -*-
"""
第8章（下）申请反欺诈：网络特征 + 不平衡采样 + LightGBM 贝叶斯调参
====================================================================
教材主 notebook 后半：把关系网络指标并入申请特征，比较 5 种不平衡采样，
最后用 bayesian-optimization 给 LightGBM 调参。

关于数据：教材给的 `loan2_analysis_net.csv` 只有 16 行（另外 3 个同名变体也只有 4 KB），
直接用它合并会让建模样本几乎为空。这里改为**从 loan1_org_net.csv 自己算**同类网络指标
生成完整的 2.7 万行网络特征表，知识点和建模流程完全一致。网络指标只取纯结构量
（度、邻居平均度、连通分量规模、是否多邻居），不用邻居标签，避免把 y 泄漏进 X。

修正：
  - SMOTE 等采样前先 fillna(0)（教材在几处漏了，会报错）；
  - `sampling_strategy=0.2` 保持一致；贝叶斯优化用 bayes_opt 3.x 的
    `BayesianOptimization(...).maximize(init_points, n_iter)` 写法；
  - LightGBM 用 `verbose=-1` 关掉训练日志。

数据：data/loan2_analysis_basic.csv、loan1_org_net.csv。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import (RandomOverSampler, SMOTE, BorderlineSMOTE)
from imblearn.combine import SMOTETomek, SMOTEENN
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
CHAPTER_DIR = ""
_EXPECTED = ("loan2_analysis_basic.csv", "loan1_org_net.csv")


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
SEED_FRAUD = "id103572"


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def save(fig, name):
    p = os.path.join(CHARTS, f"8.3_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def network_features(net):
    """从关系边表生成客户级网络指标（替代教材残缺的 loan2_analysis_net.csv）。

    注意：教材的 percentage 是"邻居中欺诈标签占比"，但关系表把同一行的 flag
    同时赋给了 userid 和 connid，这个指标会把样本自身的标签带进特征，实测
    会让所有模型 AUC 变成 1.0。这里只保留纯结构量（度、邻居平均度、连通分量
    规模、是否多邻居），不碰任何标签。
    """
    G = nx.Graph()
    for r in net.itertuples(index=False):
        G.add_edge(r.userid, r.connid)
    deg = dict(G.degree())
    try:
        avg_nb = nx.average_neighbor_degree(G)
    except Exception:
        avg_nb = {n: 0 for n in G}
    comp = {}
    for c in nx.connected_components(G):
        for n in c:
            comp[n] = len(c)
    rows = []
    for u in net.userid.unique():
        rows.append({
            "userid": u,
            "degree": deg.get(u, 0),
            "average_neighbor_degree": avg_nb.get(u, 0),
            "net_size": comp.get(u, 0),
            "is_AC": int(deg.get(u, 0) > 1),
        })
    return pd.DataFrame(rows)


def main():
    basic = pd.read_csv(os.path.join(DATA, "loan2_analysis_basic.csv"))
    net = pd.read_csv(os.path.join(DATA, "loan1_org_net.csv"))
    show("申请特征 / 关系数据", f"{basic.shape} / {net.shape}")

    nf = network_features(net)
    show("网络特征表", nf.shape)
    show("网络特征描述", nf.drop(columns="userid").describe().round(3))
    df = basic.merge(nf, on="userid", how="left")
    drop = [c for c in ["Unnamed: 0", "userid", "flag"] if c in df.columns]
    feats = [c for c in df.columns if c not in drop]
    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["flag"].astype(int)
    show("建模矩阵 / 欺诈率", f"{X.shape} / {y.mean():.2%}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.3,
                                          random_state=RS, stratify=y)
    show("训练 / 测试", f"{Xtr.shape} / {Xte.shape}")

    samplers = {
        "原始不采样": None,
        "RandomOverSampler": RandomOverSampler(sampling_strategy=.2, random_state=0),
        "SMOTE": SMOTE(sampling_strategy=.2, random_state=0),
        "BorderlineSMOTE": BorderlineSMOTE(sampling_strategy=.2, random_state=0),
        "SMOTETomek": SMOTETomek(sampling_strategy=.2, random_state=0),
        "SMOTEENN": SMOTEENN(sampling_strategy=.2, random_state=0),
    }
    samples, rows = {}, []
    base = dict(n_estimators=80, learning_rate=.1, max_depth=3,
                random_state=RS, n_jobs=-1, verbose=-1)
    for name, sp in samplers.items():
        Xs, ys = (Xtr, ytr) if sp is None else sp.fit_resample(Xtr, ytr)
        samples[name] = pd.Series(ys).value_counts()
        m = LGBMClassifier(**base).fit(Xs, ys)
        p = m.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(yte, p)
        rows.append({"方案": name, "训练样本": len(ys),
                     "正类占比": round(float(np.mean(ys)), 4),
                     "AUC": metrics.auc(fpr, tpr),
                     "KS": max(np.abs(tpr - fpr))})
        print(f"  {name} 完成")
    res = pd.DataFrame(rows).set_index("方案").round(4)
    show("五种采样方案对比", res)

    # 贝叶斯优化 LightGBM（教材 C123~C127）
    def objective(n_estimators, learning_rate, max_depth, min_split_gain):
        m = LGBMClassifier(n_estimators=int(n_estimators),
                           learning_rate=float(learning_rate),
                           max_depth=int(max_depth),
                           min_split_gain=float(min_split_gain),
                           random_state=RS, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr)
        return metrics.roc_auc_score(yte, m.predict_proba(Xte)[:, 1])

    pbounds = {"n_estimators": (30, 120), "learning_rate": (0.05, 0.2),
               "max_depth": (1, 4), "min_split_gain": (0, 5)}
    opt = BayesianOptimization(objective, pbounds, random_state=RS,
                               allow_duplicate_points=True)
    opt.maximize(init_points=2, n_iter=5)
    show("贝叶斯优化最优参数 / AUC",
         f"{opt.max['params']} / {opt.max['target']:.4f}")
    best = opt.max["params"]
    final = LGBMClassifier(n_estimators=int(best["n_estimators"]),
                           learning_rate=float(best["learning_rate"]),
                           max_depth=int(best["max_depth"]),
                           min_split_gain=float(best["min_split_gain"]),
                           random_state=RS, n_jobs=-1, verbose=-1).fit(Xtr, ytr)
    p = final.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(yte, p)
    show("调参后测试集 AUC / KS",
         pd.Series({"AUC": metrics.auc(fpr, tpr),
                    "KS": max(np.abs(tpr - fpr))}).round(4))
    show("特征重要性 Top 10",
         pd.Series(final.feature_importances_, index=feats)
         .sort_values(ascending=False).head(10))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    res[["AUC", "KS"]].plot(kind="bar", ax=ax[0],
                            color=["#5aa9ff", "#ff7b7b"])
    ax[0].set(title="不平衡采样方案对比", ylim=(0, 1))
    ax[0].tick_params(axis="x", rotation=30)
    ax[1].plot(fpr, tpr, label=f"调参后 AUC={metrics.auc(fpr, tpr):.3f}")
    ax[1].plot([0, 1], [0, 1], "k--")
    ax[1].set(title="最终 LightGBM ROC", xlabel="FPR", ylabel="TPR")
    ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout()
    save(fig, "采样方案与最终ROC")


if __name__ == "__main__" or True:
    main()
