# -*- coding: utf-8 -*-
"""
5 预测类算法的参数调优 —— 合并教材 5.1 模型参数优化方法 / 5.2&3 pipeline 保险案例
==============================================================================
  5.1 四种调参方式对比：网格搜索 GridSearchCV、Halving 搜索、随机搜索
      RandomizedSearchCV、贝叶斯搜索 Optuna（TPE 采样 + LightGBM）；
  5.2 ColumnTransformer + Pipeline：数值列均值/中位数填补、类别列众数填补 +
      序数编码，接逻辑回归；用 `<step>__<param>` 语法对整个流水线调参；
  5.3 模型保存 joblib，读取后预测新进用户。

数据：data/保险数据_第一期.csv（3.9 万行 × 50 列，抽样 8000 行演示）、
      data/保险案例_新进用户.csv。
修正：教材 notebook 用 `InteractiveShell.ast_node_interactivity='all'` 显示结果 → 改显式打印；
      OrdinalEncoder 遇到未知类别加 handle_unknown；新进用户表要先去掉标签列再预测。

运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     RandomizedSearchCV, cross_val_score)
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from scipy.stats import uniform
from xgboost import XGBClassifier
import lightgbm as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('保险数据_第一期.csv', '保险案例_新进用户.csv')   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\敏捷数据挖掘复现\5参数调优"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '5参数调优'),
              os.path.join(os.getcwd(), '敏捷数据挖掘复现', '5参数调优')]
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

SAMPLE_ROWS = 8000
RS = 42


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def save(fig, name):
    p = os.path.join(CHARTS, f"5_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def load_data():
    df = pd.read_csv(os.path.join(DATA, "保险数据_第一期.csv"))
    show("保险数据规模 / 响应率", f"{df.shape} / {df.resp_flag.mean():.3f}")
    if SAMPLE_ROWS:
        df = df.sample(n=SAMPLE_ROWS, random_state=RS).reset_index(drop=True)
    y = df.pop("resp_flag").astype(int)
    cat = df.select_dtypes(include="object").columns.tolist()
    num = df.select_dtypes(exclude="object").columns.tolist()
    if cat:
        df[cat] = OrdinalEncoder(handle_unknown="use_encoded_value",
                                 unknown_value=-1).fit_transform(df[cat].astype(str))
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return train_test_split(df, y, test_size=.3, stratify=y, random_state=RS), cat, num


def tuning(Xtr, Xte, ytr, yte):
    results = []
    base = dict(n_estimators=100, max_depth=3, learning_rate=.1,
                eval_metric="auc", random_state=RS, n_jobs=-1, verbosity=0)

    # 1 网格搜索
    t0 = time.time()
    grid = {"n_estimators": [80, 120], "max_depth": [3, 5],
            "subsample": [.8, .9], "colsample_bytree": [.8, .9]}
    gs = GridSearchCV(XGBClassifier(**base), grid, scoring="roc_auc",
                      cv=3, n_jobs=-1).fit(Xtr, ytr)
    results.append({"方法": "GridSearchCV", "候选数": 16,
                    "最优AUC": roc_auc_score(yte, gs.predict_proba(Xte)[:, 1]),
                    "耗时秒": time.time() - t0})
    show("网格搜索最优参数", gs.best_params_)

    # 2 Halving 搜索
    t0 = time.time()
    hs = HalvingGridSearchCV(XGBClassifier(**base), grid, scoring="roc_auc",
                             cv=3, factor=2, n_jobs=-1,
                             min_resources=200).fit(Xtr, ytr)
    results.append({"方法": "HalvingGridSearchCV", "候选数": 16,
                    "最优AUC": roc_auc_score(yte, hs.predict_proba(Xte)[:, 1]),
                    "耗时秒": time.time() - t0})
    show("Halving 搜索最优参数", hs.best_params_)

    # 3 随机搜索
    t0 = time.time()
    dist = {"n_estimators": [80, 100, 120, 150], "max_depth": [3, 4, 5],
            "subsample": uniform(.5, .5), "colsample_bytree": uniform(.5, .5)}
    rs = RandomizedSearchCV(XGBClassifier(**base), dist, n_iter=8,
                            scoring="roc_auc", cv=3, random_state=RS,
                            n_jobs=-1).fit(Xtr, ytr)
    results.append({"方法": "RandomizedSearchCV", "候选数": 8,
                    "最优AUC": roc_auc_score(yte, rs.predict_proba(Xte)[:, 1]),
                    "耗时秒": time.time() - t0})
    show("随机搜索最优参数", rs.best_params_)

    # 4 Optuna 贝叶斯搜索
    t0 = time.time()

    def objective(trial):
        params = {"objective": "binary", "metric": "auc", "verbosity": -1,
                  "lambda_l2": trial.suggest_float("lambda_l2", .01, 10.0),
                  "num_leaves": trial.suggest_int("num_leaves", 8, 64),
                  "learning_rate": trial.suggest_float("learning_rate", .02, .2),
                  "n_estimators": trial.suggest_int("n_estimators", 60, 160)}
        m = lgb.LGBMClassifier(**params, random_state=RS, n_jobs=-1)
        return cross_val_score(m, Xtr, ytr, cv=3, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RS))
    study.optimize(objective, n_trials=10, show_progress_bar=False)
    best_lgb = lgb.LGBMClassifier(objective="binary", metric="auc", verbosity=-1,
                                  **study.best_params, random_state=RS, n_jobs=-1)
    best_lgb.fit(Xtr, ytr)
    results.append({"方法": "Optuna(TPE)", "候选数": 10,
                    "最优AUC": roc_auc_score(yte, best_lgb.predict_proba(Xte)[:, 1]),
                    "耗时秒": time.time() - t0})
    show("Optuna 最优参数", study.best_params)

    summary = pd.DataFrame(results)
    show("四种调参方式对比", summary.round(4).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(summary.方法, summary.最优AUC, color="#5aa9ff")
    ax[0].set(title="各调参方法测试集 AUC", ylabel="AUC", ylim=(.5, .8))
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].bar(summary.方法, summary.耗时秒, color="#8b7cff")
    ax[1].set(title="耗时对比", ylabel="秒")
    ax[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save(fig, "调参方法对比")


def pipeline_case(Xtr, Xte, ytr, yte, cat, num):
    """教材 5.2：ColumnTransformer + Pipeline + 网格搜索 + 模型持久化。"""
    # 这里 Xtr 是已经编码好的 DataFrame，重新读原始表来演示 Pipeline 的自动处理
    raw = pd.read_csv(os.path.join(DATA, "保险数据_第一期.csv"))
    if SAMPLE_ROWS:
        raw = raw.sample(n=SAMPLE_ROWS, random_state=RS).reset_index(drop=True)
    labels = raw.pop("resp_flag").astype(int)
    cat_cols = raw.select_dtypes(include="object").columns.tolist()
    num_cols = raw.select_dtypes(exclude="object").columns.tolist()
    a, b, c, d = train_test_split(raw, labels, test_size=.3,
                                  stratify=labels, random_state=RS)

    pre = ColumnTransformer([
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                 unknown_value=-1))]), cat_cols),
        ("num", SimpleImputer(strategy="mean"), num_cols),
    ])
    pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=1000))])
    grid = {"lr__C": [0.1, 1.0, 10.0],
            "lr__class_weight": [None, "balanced"],
            "pre__num__strategy": ["mean", "median"]}
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=3, n_jobs=-1)
    gs.fit(a, c)
    show("Pipeline 最优参数", gs.best_params_)
    show("Pipeline 分类报告", classification_report(d, gs.predict(b), digits=3))
    show("Pipeline 测试集 AUC",
         round(roc_auc_score(d, gs.predict_proba(b)[:, 1]), 4))

    model_path = os.path.join(CHARTS, "insurance_pipeline.joblib")
    joblib.dump(gs.best_estimator_, model_path)
    reloaded = joblib.load(model_path)

    new = pd.read_csv(os.path.join(DATA, "保险案例_新进用户.csv"))
    if "resp_flag" in new.columns:
        new = new.drop(columns=["resp_flag"])
    proba = reloaded.predict_proba(new)[:, 1]
    show("新进用户预测概率", pd.Series(proba.round(4), name="响应概率"))
    print(f"  模型已保存：{model_path}")


def main():
    print("5 参数调优与 Pipeline")
    (Xtr, Xte, ytr, yte), cat, num = load_data()
    tuning(Xtr, Xte, ytr, yte)
    pipeline_case(Xtr, Xte, ytr, yte, cat, num)


if __name__ == "__main__" or True:
    main()
