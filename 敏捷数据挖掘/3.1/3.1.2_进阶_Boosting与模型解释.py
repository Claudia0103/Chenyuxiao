# -*- coding: utf-8 -*-
"""
3.1（下）进阶：XGBoost / LightGBM 原生 API 与模型解释 —— 合并 3 个进阶 notebook
==============================================================================
  XGBoost：sklearn 接口 vs 原生 DMatrix/train、参数、早停、单调约束、交互约束、
           用 num_parallel_tree 模拟随机森林；
  LightGBM：sklearn 接口 vs Dataset/train、early_stopping 回调、特征重要性、约束；
  模型解释：白盒逻辑回归系数、树模型 feature_importances_、置换重要性、
           偏依赖图(PDP)、决策路径、规则提取(export_text)。

关于 shap / dice_ml：教材自己写了「dice_ml 会把 pandas 降到 1.5，谨慎安装」，
本脚本不装这两个包，改用 sklearn.inspection 的置换重要性和偏依赖图来覆盖
"全局解释 + 单特征影响 + 决策路径 + 规则提取"这些知识点，效果等价且不破坏环境。

数据：data/进阶内容_组合算法最近特性和模型解释/保险数据_第一期.csv（3.9 万行 × 50 列，
抽样 8000 行演示，SAMPLE_ROWS 可改 None 跑全量）。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import xgboost as xgb
import lightgbm as lgb

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('保险数据_第一期.csv', '进阶内容_组合算法最近特性和模型解释')   # 该脚本会用到的 data/ 下的文件或子目录

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
DATA = os.path.join(_HERE, "data", "进阶内容_组合算法最近特性和模型解释")
CHARTS = os.path.join(_HERE, "charts")
os.makedirs(CHARTS, exist_ok=True)

SAMPLE_ROWS = 8000
RS = 42


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def save(fig, name):
    p = os.path.join(CHARTS, f"3.1.2_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def main():
    df = pd.read_csv(os.path.join(DATA, "保险数据_第一期.csv"))
    show("保险数据规模 / 响应率", f"{df.shape}，resp_flag 均值 {df.resp_flag.mean():.3f}")
    if SAMPLE_ROWS:
        df = df.sample(n=SAMPLE_ROWS, random_state=RS).reset_index(drop=True)
    y = df.pop("resp_flag").astype(int)
    cat = df.select_dtypes(include="object").columns.tolist()
    num = df.select_dtypes(exclude="object").columns.tolist()
    if cat:
        df[cat] = OrdinalEncoder(handle_unknown="use_encoded_value",
                                 unknown_value=-1).fit_transform(df[cat].astype(str))
    # sklearn 1.9 的偏依赖图拒绝整数 dtype 的列，统一转 float
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=.3,
                                          stratify=y, random_state=RS)
    show("训练 / 测试", f"{Xtr.shape} / {Xte.shape}")
    show("示例特征", list(df.columns[:10]))

    # ---------- 1 XGBoost ----------
    sk = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=.1,
                           eval_metric="auc", random_state=RS, n_jobs=-1)
    sk.fit(Xtr, ytr)
    show("XGB sklearn 接口 AUC",
         round(roc_auc_score(yte, sk.predict_proba(Xte)[:, 1]), 4))

    dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=list(Xtr.columns))
    dte = xgb.DMatrix(Xte, label=yte, feature_names=list(Xte.columns))
    params = {"max_depth": 4, "eta": .2, "objective": "binary:logistic",
              "eval_metric": "auc"}
    booster = xgb.train(params, dtr, num_boost_round=80,
                        evals=[(dtr, "train"), (dte, "test")],
                        early_stopping_rounds=8, verbose_eval=False)
    show("XGB 原生 API（含早停）AUC / 最优轮次",
         f"{roc_auc_score(yte, booster.predict(dte)):.4f} / {booster.best_iteration}")

    mono = dict(params, tree_method="exact",
                monotone_constraints=tuple([0] * len(Xtr.columns)))
    bm = xgb.train(mono, dtr, num_boost_round=30, verbose_eval=False)
    show("XGB 单调约束（全部不约束，仅演示参数）AUC",
         round(roc_auc_score(yte, bm.predict(dte)), 4))

    # DMatrix 带 feature_names 时，交互约束也要用特征名（传索引会报错）
    names = list(Xtr.columns)
    inter = dict(params, interaction_constraints=[names[:5], names[5:]])
    bi = xgb.train(inter, dtr, num_boost_round=30, verbose_eval=False)
    show("XGB 交互约束 AUC", round(roc_auc_score(yte, bi.predict(dte)), 4))

    rf_params = {"colsample_bynode": .8, "learning_rate": 1,
                 "num_parallel_tree": 20, "subsample": .8,
                 "objective": "binary:logistic", "eval_metric": "auc"}
    brf = xgb.train(rf_params, dtr, num_boost_round=1, verbose_eval=False)
    show("XGB 模拟随机森林 AUC", round(roc_auc_score(yte, brf.predict(dte)), 4))

    # ---------- 2 LightGBM ----------
    lsk = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=.1,
                             random_state=RS, n_jobs=-1, verbose=-1)
    lsk.fit(Xtr, ytr)
    show("LGB sklearn 接口 AUC",
         round(roc_auc_score(yte, lsk.predict_proba(Xte)[:, 1]), 4))
    ltr = lgb.Dataset(Xtr, label=ytr)
    lte = lgb.Dataset(Xte, label=yte, reference=ltr)
    lparams = {"objective": "binary", "metric": "auc", "num_leaves": 31,
               "learning_rate": .1, "verbosity": -1}
    lm = lgb.train(lparams, ltr, valid_sets=[lte], num_boost_round=100,
                   callbacks=[lgb.early_stopping(8, verbose=False)])
    show("LGB 原生 API（含早停）AUC / 最优轮次",
         f"{roc_auc_score(yte, lm.predict(Xte)):.4f} / {lm.best_iteration}")
    imp = pd.Series(lm.feature_importance(importance_type="split"),
                    index=Xtr.columns).sort_values(ascending=False)
    show("LGB 特征重要性(split) Top 10", imp.head(10))

    # ---------- 3 模型解释 ----------
    # 白盒模型：statsmodels 逻辑回归系数
    lr = sm.Logit(ytr, sm.add_constant(Xtr.astype(float))).fit(disp=0, maxiter=100)
    show("白盒逻辑回归系数（绝对值 Top 8）",
         lr.params.drop("const").abs().sort_values(ascending=False).head(8).round(4))

    dt = DecisionTreeClassifier(max_depth=4, random_state=RS).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=30, max_depth=6,
                                random_state=RS, n_jobs=-1).fit(Xtr, ytr)
    cmp = pd.DataFrame({
        "决策树": dt.feature_importances_,
        "随机森林": rf.feature_importances_,
        "LGB(split 归一)": imp.reindex(Xtr.columns).fillna(0).values / max(imp.sum(), 1),
    }, index=Xtr.columns)
    show("三种模型的特征重要性 Top 8（按随机森林排序）",
         cmp.sort_values("随机森林", ascending=False).head(8).round(4))

    perm = permutation_importance(rf, Xte, yte, n_repeats=3,
                                  random_state=RS, n_jobs=-1)
    show("置换重要性 Top 8",
         pd.Series(perm.importances_mean, index=Xtr.columns)
         .sort_values(ascending=False).head(8).round(4))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pd.Series(perm.importances_mean, index=Xtr.columns) \
        .sort_values().tail(10).plot(kind="barh", ax=ax, color="#8b7cff")
    ax.set(title="置换重要性 Top 10", xlabel="AUC 下降量")
    fig.tight_layout()
    save(fig, "置换重要性")

    # 偏依赖：挑两个重要特征
    feats = list(cmp.sort_values("随机森林", ascending=False).head(2).index)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    PartialDependenceDisplay.from_estimator(rf, Xte, feats, ax=ax,
                                            subsample=500, grid_resolution=20)
    fig.tight_layout()
    save(fig, "偏依赖图")

    # 决策路径
    sample = Xte.iloc[0].values.reshape(1, -1)
    nodes = dt.decision_path(sample).indices
    feat, thr = dt.tree_.feature, dt.tree_.threshold
    path = [f"节点{n}: {Xtr.columns[feat[n]]} <= {thr[n]:.2f} "
            f"(样本值 {sample[0, feat[n]]:.2f})"
            for n in nodes if feat[n] != -2]
    show("单个样本的树决策路径", "\n".join(path[:8]))

    # 规则提取：用重要性前 5 个特征重训浅树并导出文本规则
    top5 = cmp.sort_values("随机森林", ascending=False).head(5).index.tolist()
    rule_tree = DecisionTreeClassifier(max_depth=3, random_state=RS).fit(Xtr[top5], ytr)
    show("规则提取（export_text 前 12 行）",
         "\n".join(export_text(rule_tree, feature_names=top5,
                               max_depth=3).splitlines()[:12]))


if __name__ == "__main__" or True:
    main()
