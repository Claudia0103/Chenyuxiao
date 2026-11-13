# -*- coding: utf-8 -*-
"""
2.4 交叉销售（关联规则与序列模式）—— 合并教材 2.4.1 / 2.4.2 / mlxtend 上课版 / 课后答案版
====================================================================================
  1) Apriori：内联教材 myscript/apriori.py 的实现（createC1 → scanD →
     aprioriGen → arules），自己算支持度、置信度、提升度；
  2) 用 mlxtend 现成包做同一件事（TransactionEncoder + apriori + association_rules
     + fpgrowth），并回答教材两道练习题：哪些规则与「财务通」/「好享贷」有关；
  3) 序列模式：教材用 myscript 的 aprioriAll.py / prefixspan.py，这里用等价的
     「客户内按订单展开 → 枚举连续子序列 → 客户级支持度」实现，演示同样的知识点。

修正：教材 mlxtend 版写死了作者 Mac 的绝对路径 → 改 data/；`df.applymap` 在
pandas 3 已移除 → 改 `df.map`；Orange3-Associate 需要额外编译 → 用 mlxtend.fpgrowth。

数据：data/bank_prod_order.csv（GBK）、data/关联规则-使用现成的python包/bank_prod_order(1).csv。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import itertools
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori as mlx_apriori, association_rules, fpgrowth

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('bank_prod_order.csv', '关联规则-使用现成的python包')   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\敏捷数据挖掘复现\2.4交叉销售"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '2.4交叉销售'),
              os.path.join(os.getcwd(), '敏捷数据挖掘复现', '2.4交叉销售')]
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


def save(fig, name):
    p = os.path.join(CHARTS, f"2.4_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


# ---------------- 教材自带 apriori.py 的实现（内联，保证单文件自包含） ----------------
def create_c1(dataset):
    return [frozenset([i]) for i in set(itertools.chain(*dataset))]


def scan_d(dataset, candidates, min_support):
    cnt = Counter()
    for trans in dataset:
        for cand in candidates:
            if cand.issubset(trans):
                cnt[cand] += 1
    n = len(dataset)
    return {k: v / n for k, v in cnt.items() if v / n >= min_support}


def apriori_gen(Lk):
    k = len(next(iter(Lk))) if Lk else 0
    if len(Lk) > 1 and k > 0:
        L = list(Lk)
        return {L[i] | L[j]
                for i in range(len(L) - 1) for j in range(i + 1, len(L))
                if len(L[i] | L[j]) == k + 1}
    return set()


def apriori_itemsets(dataset, min_support=0.02):
    L1 = scan_d(dataset, create_c1(dataset), min_support)
    levels, current = [L1], L1
    while len(current) > 1:
        cand = apriori_gen(list(current.keys()))
        nxt = scan_d(dataset, cand, min_support) if cand else {}
        if not nxt:
            break
        levels.append(nxt)
        current = nxt
    out = {}
    for lv in levels:
        out.update(lv)
    return out


def arules(dataset, min_support=0.02):
    L = apriori_itemsets(dataset, min_support)
    rows = []
    for itemset in L:
        if len(itemset) < 2:
            continue
        items = list(itemset)
        for k in range(1, len(items)):
            for lhs in itertools.combinations(items, k):
                lhs, rhs = frozenset(lhs), frozenset(set(items) - set(lhs))
                if lhs not in L or rhs not in L:
                    continue
                sup = L[itemset]
                conf = sup / L[lhs]
                rows.append({"LHS": set(lhs), "RHS": set(rhs),
                             "support": sup, "confidence": conf,
                             "lift": conf / L[rhs]})
    return pd.DataFrame(rows)


# ---------------- 序列模式（等价实现） ----------------
def frequent_sequences(customer_orders, max_len=3, min_support=0.05):
    """customer_orders: 每个客户的产品订单序列 list[list[list[str]]]。
    枚举客户内的连续子序列，按客户数算支持度。"""
    cnt, n = Counter(), len(customer_orders)
    for orders in customer_orders:
        flat = [p for order in orders for p in order]
        seen = set()
        for L in range(1, max_len + 1):
            for i in range(len(flat) - L + 1):
                seen.add(tuple(flat[i:i + L]))
        cnt.update(seen)
    rows = [{"序列": " → ".join(seq), "长度": len(seq),
             "客户数": c, "支持度": c / n}
            for seq, c in cnt.items() if c / n >= min_support]
    return (pd.DataFrame(rows).sort_values(["长度", "支持度"], ascending=[True, False])
            .reset_index(drop=True))


def main():
    data = pd.read_csv(os.path.join(DATA, "bank_prod_order.csv"), encoding="gbk")
    show("订单数据规模 / 字段", f"{data.shape} / {list(data.columns)}")
    show("产品种类", data.PROD.nunique())
    baskets = data.groupby("CSR_ID")["PROD"].apply(list)
    show("购物篮示例", baskets.head(3))

    # ---- 1 教材自带 Apriori ----
    rules = arules(baskets.tolist(), min_support=0.02)
    show("手写 Apriori 规则数", len(rules))
    conf = rules[(rules.confidence > 0.1) & (rules.lift > 1)] \
        .sort_values("support", ascending=False)
    show("过滤后 Top 10 规则", conf.head(10).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(conf.lift, conf.support, c=conf.confidence,
                    cmap="viridis", s=25, alpha=.8)
    ax.set(title="规则分布（手写 Apriori）", xlabel="lift", ylabel="support")
    fig.colorbar(sc, ax=ax, label="confidence")
    fig.tight_layout()
    save(fig, "关联规则分布")

    # ---- 2 mlxtend 版本 + FP-Growth ----
    alt = os.path.join(DATA, "关联规则-使用现成的python包", "bank_prod_order(1).csv")
    df_src = pd.read_csv(alt, encoding="gbk") if os.path.exists(alt) else data
    grouped = df_src.groupby("CSR_ID")["PROD"].apply(list)
    te = TransactionEncoder()
    encoded = pd.DataFrame(te.fit(grouped).transform(grouped).astype(int),
                           columns=te.columns_)
    show("one-hot 矩阵", encoded.shape)

    freq = mlx_apriori(encoded, min_support=0.01, use_colnames=True)
    show("mlxtend Apriori 频繁项集数（按长度）",
         freq.assign(k=freq.itemsets.map(len)).groupby("k").size())
    try:
        rules2 = association_rules(freq, metric="lift", min_threshold=1,
                                   num_itemsets=len(encoded))
    except TypeError:
        rules2 = association_rules(freq, metric="lift", min_threshold=1)
    show("mlxtend 关联规则数", len(rules2))
    top = rules2[(rules2.confidence > 0.1) & (rules2.lift > 1)] \
        .sort_values("support", ascending=False)
    show("Top 10 规则（mlxtend）",
         top[["antecedents", "consequents", "support", "confidence", "lift"]]
         .head(10).to_string(index=False))

    for kw in ("财务通", "好享贷"):
        hit = rules2[rules2.consequents.astype(str).str.contains(kw)
                     | rules2.antecedents.astype(str).str.contains(kw)]
        show(f"教材练习题：与「{kw}」相关的规则（Top 5）",
             hit[["antecedents", "consequents", "support", "confidence", "lift"]]
             .sort_values("lift", ascending=False).head(5).to_string(index=False))

    fp = fpgrowth(encoded, min_support=0.01, use_colnames=True)
    show("FP-Growth 与 Apriori 频繁项集数是否一致",
         f"{len(fp)} vs {len(freq)}")

    # ---- 3 序列模式 ----
    by_order = data.groupby(["OrderNumber", "CSR_ID"])["PROD"].apply(list)
    customer_orders = by_order.groupby(level=1).apply(list).tolist()
    show("客户数 / 序列示例", f"{len(customer_orders)}，示例：{customer_orders[0][:2]}")
    seq = frequent_sequences(customer_orders, max_len=3, min_support=0.05)
    show("频繁序列模式 Top 15（长度≤3）", seq.head(15).to_string(index=False))
    show("长度分布", seq.groupby("长度").size())


if __name__ == "__main__" or True:
    main()
