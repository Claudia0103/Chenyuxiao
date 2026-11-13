# -*- coding: utf-8 -*-
"""
补充：社交网络（下）—— 合并 04 / 04-子网分割 / 05 三个 notebook
================================================================
  04 用《冰与火之歌》人物关系边表建图并绘制；
  04-子网分割 / 05 用 Louvain（python-louvain）做社区发现，按社区给节点着色，
  并在银行交易对手明细上找交易圈。

修正：
  - `import community; community.best_partition` 在不同版本写法不一 → 兼容函数；
  - `matplotlib.cm.get_cmap` 在 matplotlib 3.11 已移除 → `plt.get_cmap`；
  - Windows 路径 `.\data\stormofswords.csv` 改成相对 data/ 的绝对路径定位；
  - 银行交易圈原表 6903 行全量跑 Louvain 可以，但为画图清晰只展示前 6 大社区。

数据：data/stormofswords.csv、data/Archive-社交网络-交易对手明细汇总.csv。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
CHAPTER_DIR = ""
_EXPECTED = ("stormofswords.csv", "Archive-社交网络-交易对手明细汇总.csv")


def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\金融数据分析案例实战复现\补充-社交网络"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(), os.path.join(os.getcwd(), "补充-社交网络"),
              os.path.join(os.getcwd(), "金融数据分析案例实战复现", "补充-社交网络")]
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
    p = os.path.join(CHARTS, f"补2_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def best_partition(G):
    try:
        import community.community_louvain as cl
        return cl.best_partition(G, random_state=42)
    except Exception:
        import community
        fn = getattr(community, "best_partition", None) or \
             getattr(community, "community_louvain").best_partition
        return fn(G)


def main():
    edges = pd.read_csv(os.path.join(DATA, "stormofswords.csv"))
    show("人物关系边表", f"{edges.shape} / {list(edges.columns)}")
    edges["Weight"] = pd.to_numeric(edges.Weight, errors="coerce").fillna(1).astype(int)
    G = nx.from_pandas_edgelist(edges, "Source", "Target",
                                edge_attr="Weight", create_using=nx.Graph())
    show("图规模（节点/边）", f"{G.number_of_nodes()} / {G.number_of_edges()}")
    show("度最高的 5 个角色", sorted(G.degree, key=lambda x: -x[1])[:5])
    show("连通分量数", nx.number_connected_components(G))

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=5, k=.35)
    nx.draw_networkx(G, pos, node_size=45, node_color="#5aa9ff",
                     with_labels=False, alpha=.8, ax=ax)
    ax.set_title("《冰与火之歌》人物关系网络"); ax.axis("off")
    fig.tight_layout()
    save(fig, "人物关系网络")

    part = best_partition(G)
    part_s = pd.Series(part, name="group")
    show("Louvain 社区数 / 规模 Top 6",
         f"{part_s.nunique()} / {part_s.value_counts().head(6).to_dict()}")
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=55, ax=ax,
                           node_color=[part[n] for n in G.nodes], cmap=cmap)
    nx.draw_networkx_edges(G, pos, alpha=.25, ax=ax)
    ax.set_title("Louvain 社区着色（人物关系网）"); ax.axis("off")
    fig.tight_layout()
    save(fig, "人物关系社区")

    # 子网分割：取最大社区单独画
    biggest = int(part_s.value_counts().idxmax())
    nodes = [n for n, g in part.items() if g == biggest]
    sub = G.subgraph(nodes)
    show(f"最大社区（编号 {biggest}）规模", f"{sub.number_of_nodes()} 节点 / "
         f"{sub.number_of_edges()} 边")
    fig, ax = plt.subplots(figsize=(7, 5))
    nx.draw_networkx(sub, nx.spring_layout(sub, seed=2), node_size=180,
                     node_color="#8b7cff", with_labels=True, font_size=7, ax=ax)
    ax.set_title(f"最大社区子网（编号 {biggest}）"); ax.axis("off")
    fig.tight_layout()
    save(fig, "最大社区子网")

    # 银行交易圈实践（教材 05）
    bank = pd.read_csv(os.path.join(DATA, "Archive-社交网络-交易对手明细汇总.csv"),
                       index_col=0)
    show("交易对手明细", f"{bank.shape} / {list(bank.columns)[:6]}")
    keep = ["源客户号", "目标客户号"]
    bank = bank[keep].dropna()
    B = nx.from_pandas_edgelist(bank, "源客户号", "目标客户号")
    show("交易网络规模", f"{B.number_of_nodes()} 节点 / {B.number_of_edges()} 边")
    bp = best_partition(B)
    bs = pd.Series(bp, name="group")
    show("交易圈（社区）数 / 规模 Top 6",
         f"{bs.nunique()} / {bs.value_counts().head(6).to_dict()}")
    show("最大交易圈占全部节点比例",
         f"{bs.value_counts().max() / len(bs):.2%}")

    fig, ax = plt.subplots(figsize=(7, 4))
    bs.value_counts().head(6).plot(kind="bar", ax=ax, color="#3ddc97")
    ax.set(title="银行交易圈规模 Top 6", xlabel="社区编号", ylabel="客户数")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    save(fig, "交易圈规模")


if __name__ == "__main__" or True:
    main()
