# -*- coding: utf-8 -*-
"""
3.2 违规识别（无监督异常检测）—— 由《3.2违规识别(IsolationForrest_LOF无监督).ipynb》整理
====================================================================================
信用卡交易欺诈识别，教材用两个无监督算法，并用 ARI（调整兰德指数）对照真实标签：
  孤立森林 IsolationForest（contamination 取坏样本比例的 5~20 倍）
  → 随机搜索参数 → 局部异常因子 LOF。

数据：教材 creditcard.csv 有 147 MB / 28 万行；复现目录里放的是精简样本
data/creditcard_sample.csv（前 3 万行 + 全部 473 条欺诈共 30258 行），
方法完全一致、跑得快。想用全量数据把原文件放回 data/ 并改 DATA_FILE 即可。

修正：教材用 RandomizedSearchCV 对无监督模型调参需要自定义 scorer，容易踩坑，
这里改成显式遍历少量参数组合、按 ARI 选最优，结果更可解释；LOF 的
n_neighbors 从 500 降到 50（3 万行下 500 邻居会非常慢）。

运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import adjusted_rand_score, roc_auc_score

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ()   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\敏捷数据挖掘复现\3.2违规识别"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '3.2违规识别'),
              os.path.join(os.getcwd(), '敏捷数据挖掘复现', '3.2违规识别')]
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

DATA_FILE = "creditcard_sample.csv"
RS = 42


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def save(fig, name):
    p = os.path.join(CHARTS, f"3.2_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def report(name, y, pred, score=None):
    binary = np.where(pred == -1, 1, 0)
    ari = adjusted_rand_score(y, pred)
    out = {"ARI": ari, "检出异常数": int((pred == -1).sum()),
           "其中真欺诈": int(((binary == 1) & (y == 1)).sum())}
    if score is not None:
        out["异常分数AUC"] = roc_auc_score(y, score)
    show(name, pd.Series(out).round(4))
    show(f"{name} 混淆（行=真实，列=-1异常/1正常）",
         pd.crosstab(y, pred))
    return out


def main():
    data = pd.read_csv(os.path.join(DATA, DATA_FILE))
    show("数据规模 / 欺诈率", f"{data.shape}，Class=1 占比 {data.Class.mean():.3%}")
    y = data["Class"].values
    vcols = [c for c in data.columns if c.startswith("V")]
    X = data[vcols]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    data.Class.value_counts().sort_index().plot(kind="bar", ax=ax[0],
                                                color=["#5aa9ff", "#ff7b7b"])
    ax[0].set(title="类别分布", xlabel="Class", ylabel="样本数")
    ax[0].tick_params(axis="x", rotation=0)
    X.kurtosis().sort_values().plot(ax=ax[1], color="#8b7cff")
    ax[1].set(title="各变量峰度（孤立森林按峰度采样）", xlabel="变量", ylabel="峰度")
    ax[1].tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    save(fig, "数据分布与峰度")
    show("峰度最高的 5 个变量", X.kurtosis().sort_values(ascending=False).head(5).round(2))

    # ---- 1 孤立森林 ----
    base_rate = data.Class.mean()
    contamination = min(.1, max(.01, base_rate * 10))
    show("按教材规则取 contamination（坏样本比例 10 倍）", round(contamination, 4))
    iso = IsolationForest(contamination=contamination, n_estimators=100,
                          max_samples=.8, random_state=RS, n_jobs=-1)
    iso.fit(X)
    pred_iso = iso.predict(X)
    score_iso = -iso.score_samples(X)          # 越大越异常
    iso_out = report("孤立森林（默认参数）", y, pred_iso, score_iso)

    # ---- 2 显式遍历参数（等价教材的随机搜索） ----
    rows = []
    for cont in (0.01, 0.02, 0.035):
        for n_est in (80, 120):
            m = IsolationForest(contamination=cont, n_estimators=n_est,
                                max_samples=.8, random_state=RS, n_jobs=-1).fit(X)
            p = m.predict(X)
            rows.append({"contamination": cont, "n_estimators": n_est,
                         "ARI": adjusted_rand_score(y, p)})
    tune = pd.DataFrame(rows).sort_values("ARI", ascending=False)
    show("参数遍历结果（按 ARI）", tune.round(4).to_string(index=False))
    best = tune.iloc[0]
    best_iso = IsolationForest(contamination=float(best.contamination),
                               n_estimators=int(best.n_estimators),
                               max_samples=.8, random_state=RS, n_jobs=-1).fit(X)
    best_out = report("孤立森林（调参后）", y, best_iso.predict(X),
                      -best_iso.score_samples(X))

    # ---- 3 局部异常因子 ----
    t0 = time.time()
    lof = LocalOutlierFactor(n_neighbors=50, contamination=contamination,
                             novelty=True, n_jobs=-1)
    lof.fit(X)
    pred_lof = lof.predict(X)
    score_lof = -lof.score_samples(X)
    lof_out = report("LOF 局部异常因子", y, pred_lof, score_lof)
    print(f"  LOF 耗时 {time.time() - t0:.1f}s")

    # ---- 4 汇总与异常分数分布 ----
    summary = pd.DataFrame({"孤立森林": iso_out, "孤立森林调参": best_out,
                            "LOF": lof_out}).T
    show("三种方案汇总", summary)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for s, lab, c in [(score_iso, "IsolationForest", "#5aa9ff"),
                      (score_lof, "LOF", "#ff7b7b")]:
        ax[0].hist(s[y == 0], bins=60, alpha=.5, label=f"{lab} 正常", color=c,
                   density=True)
        ax[0].hist(s[y == 1], bins=60, alpha=.5, label=f"{lab} 欺诈",
                   color=c, histtype="step", density=True, lw=2)
    ax[0].set(title="异常分数分布", xlabel="异常分数", ylabel="密度")
    ax[0].legend(fontsize=8)
    summary.ARI.plot(kind="bar", ax=ax[1], color="#8b7cff")
    ax[1].set(title="ARI 对比", ylabel="调整兰德指数")
    ax[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    save(fig, "异常分数与ARI对比")


if __name__ == "__main__" or True:
    main()
