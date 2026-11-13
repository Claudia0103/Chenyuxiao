# -*- coding: utf-8 -*-
"""
第8章 线性回归 —— 由《线性回归.ipynb》整理成 .py
================================================
教材用信用卡消费数据走了一遍完整建模流程：
  清洗（删无用列 / 去重 / 填补缺失 / 编码 / 异常值 / 哑变量）
  → 相关分析 → 模型 1 → VIF 共线性 → 模型 2 → 残差诊断
  → 对数变换模型 3 → 二次项模型 4 → 交互项模型 5。

对教材代码的修正：
  1. `data_raw['Age'].iloc[40] = ...` 链式赋值在 pandas 3 里不生效 → 用 .loc；
  2. `pd.get_dummies` 新版返回 bool 列，直接进 ols 公式会出问题 → astype(int)；
  3. edu_class 里有 NaN，教材的 label.index(x) 遇到 NaN 会报错 → 用字典映射，
     缺失单独编成一个类别（这样正好得到教材公式里的 edu_1~edu_4）；
  4. 教材用 `IPython...ast_node_interactivity='all'` 显示多个结果 → 改成打印。

数据：data/LR_practice.xlsx（76 行）。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('LR_practice.xlsx',)   # 该脚本会用到的 data/ 下的文件或子目录

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


def vif(df, y_col):
    """把 y_col 当作因变量、其余列当自变量做回归，返回 1/(1-R²)。"""
    x_cols = [c for c in df.columns if c != y_col]
    r2 = ols(f"{y_col} ~ " + " + ".join(x_cols), df).fit().rsquared
    return 1. / (1. - r2)


def clean_data():
    raw = pd.read_excel(os.path.join(DATA, "LR_practice.xlsx"))
    show("原始规模", raw.shape)
    raw = raw.drop(["id", "Acc", "edad2"], axis=1).drop_duplicates().reset_index(drop=True)
    show("删列去重后", raw.shape)
    show("各列缺失比例", raw.isna().mean().round(4))

    raw["avg_exp"] = raw["avg_exp"].fillna(raw["avg_exp"].mean())
    show("avg_exp 填补后缺失比例", round(raw.avg_exp.isna().mean(), 4))

    # 分类变量编码：教材用 unique().index()，这里用字典，NaN 单列一类
    raw["gender"] = raw["gender"].map({"Male": 1, "Female": 0}).astype("float")
    labels = [x for x in raw.edu_class.dropna().unique().tolist()]
    code = {v: i for i, v in enumerate(labels)}
    raw["edu_class"] = (raw.edu_class.map(code).fillna(len(labels)).astype(int))
    show("edu_class 编码对照", pd.Series(code, name="编码"))

    # 3 倍标准差找异常年龄（教材里是 999，位于第 40 行）
    # pandas 3 严格校验 dtype，int 列写不进小数均值，先转 float
    raw["Age"] = raw["Age"].astype(float)
    z = np.asarray(stats.zscore(raw["Age"]), dtype=float)
    bad = (z > 3) | (z < -3)
    show("3σ 检出的异常年龄", raw.loc[bad, "Age"].tolist())
    fill = raw.loc[~bad, "Age"].mean()
    raw.loc[bad, "Age"] = fill
    show(f"用其余样本均值填补：{fill:.2f}", raw.loc[bad, "Age"].round(2).tolist())

    dummy = pd.get_dummies(raw["edu_class"], prefix="edu", drop_first=True).astype(int)
    data = pd.concat([raw, dummy], axis=1)
    show("哑变量列名（对应教材公式 edu_1~edu_4）", list(dummy.columns))
    return data


def main():
    data = clean_data()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.boxplot(y=data["dist_avg_income"], ax=ax[0], color="#5aa9ff")
    ax[0].set(title="dist_avg_income 箱线图", ylabel="取值")
    ax[1].scatter(data["Income"], data["avg_exp"], s=25, color="#ff7b7b")
    ax[1].set(title="Income 与 avg_exp 散点", xlabel="Income", ylabel="avg_exp")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    p0 = os.path.join(CHARTS, "8.3_清洗诊断与散点.png")
    fig.savefig(p0, dpi=120); plt.close(fig)

    show("Pearson 相关系数（部分）",
         data[["avg_exp", "Age", "Income", "dist_home_val", "dist_avg_income"]]
         .corr().round(3))
    show("Kendall 相关系数（部分）",
         data[["avg_exp", "Age", "Income", "dist_home_val"]]
         .corr(method="kendall").round(3))

    # ---------- 五个模型 ----------
    base = ("gender + Age + Income + Ownrent + Selfempl + dist_home_val + "
            "edu_1 + edu_2 + edu_3 + edu_4")
    f1 = f"avg_exp ~ {base} + dist_avg_income"
    f2 = f"avg_exp ~ {base}"
    f3 = f"ln_avg_exp ~ {base}"
    f4 = f"ln_avg_exp ~ {base} + Age_sq"
    f5 = f"ln_avg_exp ~ {base} + Age_sq + Age_gender"

    m1 = ols(f1, data).fit()
    show("模型 1 summary（教材 ols 结果）", m1.summary())

    exog = data[["gender", "Age", "Income", "Ownrent", "Selfempl",
                 "dist_home_val", "dist_avg_income", "edu_class"]]
    show("VIF 共线性检查", pd.Series({c: vif(exog, c) for c in exog.columns}).round(3))
    print("  dist_avg_income 与 dist_home_val 高度相关，模型 2 去掉 dist_avg_income")

    m2 = ols(f2, data).fit()
    show("模型 2 summary", m2.summary())

    data["ln_avg_exp"] = np.log(data["avg_exp"])
    m3 = ols(f3, data).fit()
    data["Age_sq"] = data["Age"] ** 2
    m4 = ols(f4, data).fit()
    data["Age_gender"] = data["Age"] * data["gender"]
    m5 = ols(f5, data).fit()

    models = {"M1 原始全变量": m1, "M2 去共线变量": m2, "M3 对数因变量": m3,
              "M4 加 Age²": m4, "M5 加交互项": m5}
    show("五个模型对比", pd.DataFrame({
        "R2": [m.rsquared for m in models.values()],
        "adj_R2": [m.rsquared_adj for m in models.values()],
        "AIC": [m.aic for m in models.values()],
        "BIC": [m.bic for m in models.values()],
        "显著系数个数(|p|<0.05)": [(m.pvalues < .05).sum() for m in models.values()],
    }, index=models.keys()).round(3))

    # ---------- 残差诊断 ----------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].scatter(m2.predict(data), m2.resid, s=25, color="#5aa9ff")
    axes[0, 0].axhline(0, color="gray", lw=1)
    axes[0, 0].set(title="M2 残差 vs 拟合值", xlabel="拟合值", ylabel="残差")
    stats.probplot(m2.resid, plot=axes[0, 1])
    axes[0, 1].set_title("M2 残差 QQ 图")
    stats.probplot(m3.resid, plot=axes[1, 0])
    axes[1, 0].set_title("M3（对数变换）残差 QQ 图")
    aics = [m.aic for m in models.values()]
    axes[1, 1].bar(range(len(models)), aics, color="#8b7cff")
    axes[1, 1].set_xticks(range(len(models)))
    axes[1, 1].set_xticklabels(["M1", "M2", "M3", "M4", "M5"])
    axes[1, 1].set(title="五个模型 AIC（注意 M3+ 因变量量纲不同，不可直接比）", ylabel="AIC")
    fig.tight_layout()
    p1 = os.path.join(CHARTS, "8.3_残差诊断与模型对比.png")
    fig.savefig(p1, dpi=120); plt.close(fig)
    print(f"\n图已保存：{os.path.basename(p0)}、{os.path.basename(p1)}")


if __name__ == "__main__" or True:
    main()
