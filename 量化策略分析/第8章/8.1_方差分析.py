# -*- coding: utf-8 -*-
"""
第8章 单因素 / 多因素方差分析 —— 由《方差分析.ipynb》整理成 .py
================================================================
数据：信用卡消费 creditcard_exp.csv，看不同学历(edu_class)、性别(gender)
的平均消费 avg_exp 有没有显著差异。教材只打印了 ols(...).summary()，
这里补上 anova_lm 方差分析表和分组均值/箱线图，结论更直观。

教材代码里的坑：avg_exp 是字符串列，里面有空格 ' ' 当缺失标记，
教材用 apply(lambda x: 0 if x==' ' else x) 把它当 0 处理再转 float。
本脚本照做以便和教材结果对齐；实际项目里建议改成 NaN 后剔除，
脚本里也给出了这第二种口径的结果，方便比较。

运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('creditcard_exp.csv',)   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\量化策略分析复现\第8章-统计检验与预测"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '第8章-统计检验与预测'),
              os.path.join(os.getcwd(), '量化策略分析复现', '第8章-统计检验与预测')]
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


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def load(space_as_zero=True):
    data = pd.read_csv(os.path.join(DATA, "creditcard_exp.csv"))
    df = data[["avg_exp", "edu_class", "gender"]].dropna(how="any", axis=0).copy()
    df["edu_class"] = df["edu_class"].astype(str)
    df["gender"] = df["gender"].astype(str)
    if space_as_zero:
        df["avg_exp"] = df["avg_exp"].apply(lambda x: 0 if x == " " else x).astype(float)
    else:
        df["avg_exp"] = pd.to_numeric(df["avg_exp"].replace(" ", np.nan), errors="coerce")
        df = df.dropna(subset=["avg_exp"])
    return df


def main():
    df = load(space_as_zero=True)
    show("样本量与分组均值（教材口径：空格当 0）",
         pd.DataFrame({"样本量": [len(df)],
                       "avg_exp 均值": [df.avg_exp.mean()]}).round(2))
    show("按学历分组均值", df.groupby("edu_class").avg_exp.agg(["count", "mean", "std"]).round(2))
    show("按性别分组均值", df.groupby("gender").avg_exp.agg(["count", "mean", "std"]).round(2))

    # 单因素（教材写的是含两个主效应的模型）
    f1 = "avg_exp ~ C(edu_class) + C(gender)"
    m1 = ols(f1, data=df).fit()
    show("单因素/双主效应模型 summary", m1.summary())
    show("anova_lm 方差分析表 (typ=2)", anova_lm(m1, typ=2).round(4))

    # 多因素：加入交互项，* 等价于两个主效应 + 冒号交互项
    f2 = "avg_exp ~ C(edu_class) * C(gender)"
    m2 = ols(f2, data=df).fit()
    show("含交互项模型 summary", m2.summary())
    show("含交互项 anova_lm typ=2", anova_lm(m2, typ=2).round(4))

    # 第二种口径：空格当缺失剔除，看结论是否稳定
    df2 = load(space_as_zero=False)
    m3 = ols(f1, data=df2).fit()
    show("空格当缺失后的样本量与 R²",
         pd.Series({"样本量": len(df2), "R2": m3.rsquared,
                    "edu_class 的 F 检验 p 值": anova_lm(m3, typ=2).loc[
                        "C(edu_class)", "PR(>F)"]}).round(4))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    groups = [df.loc[df.edu_class == g, "avg_exp"].values
              for g in sorted(df.edu_class.unique())]
    ax[0].boxplot(groups, tick_labels=sorted(df.edu_class.unique()))
    ax[0].set(title="不同学历的月均消费", xlabel="edu_class", ylabel="avg_exp")
    means = df.groupby(["edu_class", "gender"]).avg_exp.mean().unstack()
    means.plot(kind="bar", ax=ax[1], color=["#5aa9ff", "#ff7b7b"])
    ax[1].set(title="学历 × 性别 分组均值", xlabel="edu_class", ylabel="avg_exp")
    ax[1].tick_params(axis="x", rotation=0)
    ax[1].legend(title="gender")
    fig.tight_layout()
    out = os.path.join(CHARTS, "8.1_分组分布与交互均值.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\n图已保存：{out}")


if __name__ == "__main__" or True:
    main()
