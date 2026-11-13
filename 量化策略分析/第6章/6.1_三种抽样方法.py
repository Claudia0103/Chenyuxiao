# -*- coding: utf-8 -*-
"""
第6章 抽样 —— 由《抽样.ipynb》和教材附带的 6_1sampling.py 整理成 .py
==================================================================
教材给了一个 get_sample 函数，支持简单随机、分层、系统三种抽样，
每种又分"按个数 k"和"按比例 k(0<k<1)"两种取法，共 6 种用法。

对教材代码的修正（老版本 Python / pandas / numpy 的写法）：
  1. `sampling is "simple_random"` 用 is 比较字符串，新 Python 会告警甚至失效 → 改成 ==；
  2. `np.in1d` 在 numpy 2.x 已弃用 → 改用集合判断；
  3. `np.NaN`、`np.in1d` 等同理；空 DataFrame 逐层 concat → 改成收集后一次 concat；
  4. 教材读的是作者本机绝对路径、且没设随机种子 → 改相对路径 + random_state，结果可复现。
  5. 系统抽样教材用 step = n // k + 1，实际抽到的条数会略少于 k，这里保留教材行为并打印实际条数。

数据：data/clients.csv（5369 条客户记录，GBK 编码，分层列 Dist）。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('clients.csv',)   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\量化策略分析复现\第6章-市场调研与数据预处理"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '第6章-市场调研与数据预处理'),
              os.path.join(os.getcwd(), '量化策略分析复现', '第6章-市场调研与数据预处理')]
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


def get_sample(df, sampling="simple_random", k=1, stratified_col=None,
               random_state=0):
    """三种抽样。k>=1 表示样本个数(必须 int)；0<k<1 表示抽样比例。

    分层抽样时，k>=1 表示每层取 k 条；0<k<1 表示每层按该比例取。
    返回保留原索引的 DataFrame。
    """
    if stratified_col is None:
        stratified_col = []
    if isinstance(stratified_col, str):
        stratified_col = [stratified_col]
    n = len(df)
    if k <= 0:
        raise ValueError("k 必须大于 0")
    if k >= 1 and not isinstance(k, (int, np.integer)):
        raise TypeError("按个数抽样时 k 必须是正整数")
    sample_by_n = k >= 1
    rng = np.random.default_rng(random_state)

    if sampling not in ("simple_random", "stratified", "systematic"):
        raise ValueError("sampling 只能是 simple_random / stratified / systematic")

    if sampling == "simple_random":
        size = k if sample_by_n else math.ceil(n * k)
        size = min(size, n)
        idx = rng.choice(n, size=size, replace=False)
        return df.iloc[sorted(idx)].copy()

    if sampling == "systematic":
        size = k if sample_by_n else math.ceil(n * k)
        step = n // size + 1                 # 教材原公式，实际条数会略少
        idx = list(range(n))[0::step]
        return df.iloc[idx].copy()

    # 分层抽样
    missing = [c for c in stratified_col if c not in df.columns]
    if missing:
        raise ValueError(f"分层列不存在：{missing}")
    if not stratified_col:
        raise ValueError("分层抽样必须传 stratified_col")
    parts = []
    for _, sub in df.groupby(stratified_col, dropna=False):
        take = k if sample_by_n else math.ceil(len(sub) * k)
        take = min(int(take), len(sub))
        if take <= 0:
            continue
        idx = rng.choice(len(sub), size=take, replace=False)
        parts.append(sub.iloc[sorted(idx)].copy())
    return pd.concat(parts) if parts else df.iloc[0:0].copy()


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def main():
    clients = pd.read_csv(os.path.join(DATA, "clients.csv"), encoding="gbk")
    show("clients.csv 规模与字段", f"{clients.shape}，字段 {list(clients.columns)}")
    show("各字段非空计数", clients.count())
    show("分层列 Dist 的层数与每层样本量", clients["Dist"].value_counts().sort_index())

    cases = [
        ("简单随机-按个数", "simple_random", 22, None),
        ("简单随机-按比例", "simple_random", 0.1, None),
        ("分层-每层按个数", "stratified", 2, ["Dist"]),
        ("分层-每层按比例", "stratified", 0.1, ["Dist"]),
        ("系统-按个数", "systematic", 4, None),
        ("系统-按比例", "systematic", 0.1, None),
    ]
    sizes = {}
    for label, method, k, col in cases:
        s = get_sample(clients, sampling=method, k=k,
                       stratified_col=col, random_state=666)
        sizes[label] = len(s)
        show(f"{label}（k={k}）",
             f"抽到 {len(s)} 条 / 总体 {len(clients)} 条，占比 {len(s)/len(clients):.2%}")
        if col:
            show(f"{label} 各层样本量", s.groupby("Dist", dropna=False).size())

    # 系统抽样用教材的 step 公式，实际条数会少于名义 k，打印出来便于对照
    show("抽样结果条数汇总", pd.Series(sizes, name="样本量"))

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(list(sizes), list(sizes.values()), color="#5aa9ff")
    ax.bar_label(bars, padding=3)
    ax.set(title="六种抽样方式的样本量对比", ylabel="样本量")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out = os.path.join(CHARTS, "6.1_六种抽样样本量对比.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\n图已保存：{out}")


if __name__ == "__main__" or True:
    main()
