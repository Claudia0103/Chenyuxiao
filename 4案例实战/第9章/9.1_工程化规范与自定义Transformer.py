# -*- coding: utf-8 -*-
"""
第9章 算法工程化 —— 由《第9章算法工程化代码通过.ipynb》整理
==========================================================
教材这一章讲代码规范：项目工程结构、自定义 Transformer、数据处理性能。
这里把三段都做成可运行的东西：
  1) 打印教材推荐的 cookiecutter-data-science 项目结构；
  2) CustomTimeTransformer（TransformerMixin）：从时间戳生成小时/月/季节/峰谷
     等特征，并接进 sklearn Pipeline 跑通；
  3) 性能对比：for / iterrows / itertuples / apply / 向量化 五种写法在
     teleco_camp.csv 上算同一指标，实测耗时。

修正：教材的 CustomTimeTransformer 只实现了 transform，这里补齐 fit /
get_feature_names_out，能直接放进 sklearn Pipeline；TransformerMixin 新版
要求 fit 返回 self。

数据：data/teleco_camp.csv。运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
CHAPTER_DIR = ""
_EXPECTED = ("teleco_camp.csv",)


def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\金融数据分析案例实战复现\第9章-算法工程化"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(), os.path.join(os.getcwd(), "第9章-算法工程化"),
              os.path.join(os.getcwd(), "金融数据分析案例实战复现", "第9章-算法工程化")]
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
    p = os.path.join(CHARTS, f"9.1_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


PROJECT_STRUCTURE = """项目根目录/
├── README.md            <- 项目说明与开发指南
├── Makefile             <- make data / make train 等命令
├── data/
│   ├── external/        <- 外部第三方数据
│   ├── interim/         <- 中间转换数据
│   ├── processed/       <- 可直接建模的数据
│   └── raw/             <- 原始数据（不可变）
├── docs/                <- 文档
├── models/              <- 训练好并序列化的模型
├── notebooks/           <- 探索用 notebook
├── src/                 <- 可复用的包与脚本
│   ├── data/            <- 数据生成/清洗
│   ├── features/        <- 特征工程
│   └── models/          <- 训练与预测
└── requirements.txt
"""


class CustomTimeTransformer(TransformerMixin):
    """教材的自定义时间特征处理器，补齐 fit / get_feature_names_out。"""

    def __init__(self, time_col="timestamp", period_dict=None, season_dict=None):
        self.time_col = time_col
        self.period_dict = period_dict or {0: "off-peak", 6: "peak",
                                           12: "off-peak", 18: "peak"}
        self.season_dict = season_dict or {m: s for m, s in zip(
            range(1, 13), ["winter", "winter", "spring", "spring", "spring",
                           "summer", "summer", "summer", "autumn", "autumn",
                           "autumn", "winter"])}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = pd.DataFrame(X).copy()
        t = pd.to_datetime(data[self.time_col])
        data["hour"] = t.dt.hour
        data["month"] = t.dt.month
        data["dayofweek"] = t.dt.dayofweek
        data["season"] = t.dt.month.map(self.season_dict)
        data["period"] = t.dt.hour.map(self.period_dict).fillna("off-peak")
        return data.drop(columns=[self.time_col])

    def get_feature_names_out(self, input_features=None):
        return np.array(["hour", "month", "dayofweek", "season", "period",
                         *[c for c in (input_features or []) if c != self.time_col]])


def perf_compare(df):
    """同一指标（ARPU×2 + 1）用五种写法算，比较耗时。"""
    d = df.head(5000).copy()
    # 原始 ARPU 是字符串列（pandas 3 的 string dtype 不能求中位数、乘 2 会变拼接）
    d["ARPU"] = pd.to_numeric(d["ARPU"], errors="coerce")
    arpu = d["ARPU"].fillna(0)
    out = {}

    def timed(name, fn):
        t0 = time.perf_counter()
        v = fn()
        out[name] = time.perf_counter() - t0
        return v

    timed("for 索引循环", lambda: [arpu.iloc[i] * 2 + 1 for i in range(len(arpu))])
    timed("iterrows", lambda: [r.ARPU * 2 + 1 if pd.notna(r.ARPU) else 1
                               for _, r in d[["ARPU"]].iterrows()])
    timed("itertuples", lambda: [r.ARPU * 2 + 1 if pd.notna(r.ARPU) else 1
                                 for r in d[["ARPU"]].itertuples()])
    timed("apply", lambda: list(arpu.apply(lambda x: x * 2 + 1)))
    timed("向量化", lambda: (arpu * 2 + 1).tolist())
    return pd.Series(out).sort_values()


def main():
    show("教材推荐的项目工程结构", PROJECT_STRUCTURE)

    demo = pd.DataFrame({
        "timestamp": ["2023-01-01 10:00:00", "2023-02-01 15:30:00",
                      "2023-07-01 07:10:00", "2023-12-01 20:00:00"],
        "col1": [10, 20, 30, 40],
    })
    tt = CustomTimeTransformer()
    show("CustomTimeTransformer 输出", tt.fit_transform(demo))

    # 接进 sklearn Pipeline（用带时间列的电信数据）
    raw = pd.read_csv(os.path.join(DATA, "teleco_camp.csv"))
    show("电信数据规模 / 字段", f"{raw.shape} / {list(raw.columns)[:10]}")
    rng = np.random.RandomState(0)
    base = pd.Timestamp("2023-01-01")
    df = raw.head(2000).copy()
    df["ARPU"] = pd.to_numeric(df["ARPU"], errors="coerce").fillna(0)
    df["timestamp"] = [base + pd.Timedelta(hours=int(h))
                       for h in rng.randint(0, 24 * 365, len(df))]
    target = (df["ARPU"] > df["ARPU"].median()).astype(int)
    X = df[["timestamp", "ARPU"]]
    Xtr, Xte, ytr, yte = train_test_split(X, target, test_size=.3,
                                          random_state=0, stratify=target)
    pipe = Pipeline([
        ("time", CustomTimeTransformer()),
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    Xtr_t = pipe.fit_transform(Xtr)
    show("Pipeline 转换后的特征", pd.DataFrame(Xtr_t).head())
    try:
        num = pd.DataFrame(Xtr_t).apply(pd.to_numeric, errors="coerce").fillna(0)
        clf = LogisticRegression(max_iter=500).fit(num, ytr)
        num_te = pd.DataFrame(pipe.transform(Xte)).apply(
            pd.to_numeric, errors="coerce").fillna(0)
        show("Pipeline + 逻辑回归 测试准确率", round(clf.score(num_te, yte), 4))
        print("  （演示标签由 ARPU 自身构造，准确率 1.0 属预期，只用于验证 Pipeline 能跑通）")
    except Exception as e:
        print("  Pipeline 建模跳过：", type(e).__name__, e)

    perf = perf_compare(raw)
    show("五种写法耗时（秒，5000 行）", perf.round(4))
    show("向量化相对 for 循环的加速比",
         round(float(perf["for 索引循环"] / max(perf["向量化"], 1e-9)), 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    perf.plot(kind="barh", ax=ax, color="#5aa9ff")
    ax.set(title="数据处理写法耗时对比", xlabel="秒")
    fig.tight_layout()
    save(fig, "写法性能对比")


if __name__ == "__main__" or True:
    main()
