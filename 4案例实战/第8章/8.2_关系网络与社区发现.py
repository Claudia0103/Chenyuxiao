# -*- coding: utf-8 -*-
"""
第8章（中）申请反欺诈：关系网络与社区发现 —— 由主 notebook 网络部分整理
======================================================================
从一个已知欺诈者 id103572 出发，沿"申请人—联系人"关系找关联客群，
构建 networkx 图，用 Louvain 做社区发现，找欺诈者聚集的社团，
再按申请日期逐日建图观察欺诈子网的演化。

修正与替代：
  - 教材 `import community; community.best_partition(G)` 在新版 python-louvain
    里写法有变化，脚本用兼容写法；
  - `matplotlib.cm.get_cmap` 在 matplotlib 3.11 已移除 → `plt.get_cmap`；
  - 全量 49799 条边跑 Louvain 较慢，默认按顺序取 NET_EDGES 条边建图，
    改成 None 可跑全量。

数据：data/loan1_org_net.csv。运行：整份代码粘进 Jupyter 一个 cell 也能跑。
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
_EXPECTED = ("loan1_org_net.csv",)


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

SEED_FRAUD = "id103572"
NET_EDGES = 8000       # 改成 None 用全量 49799 条边


def show(title, obj):
    print("\n" + "=" * 10 + " " + str(title) + " " + "=" * 10)
    print(obj)


def save(fig, name):
    p = os.path.join(CHARTS, f"8.2_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def best_partition(G):
    """兼容不同版本 python-louvain 的 community 接口。"""
    try:
        import community.community_louvain as cl
        return cl.best_partition(G, random_state=42)
    except Exception:
        import community
        fn = getattr(community, "best_partition", None) or \
             getattr(community, "community_louvain").best_partition
        return fn(G)


def build_graph(df):
    G = nx.Graph()
    for r in df.itertuples(index=False):
        G.add_edge(r.userid, r.connid)
        G.nodes[r.userid].setdefault("flag", r.flag)
        G.nodes[r.connid].setdefault("flag", r.flag)
    return G


def main():
    dat0 = pd.read_csv(os.path.join(DATA, "loan1_org_net.csv"))
    show("关系数据规模 / 字段", f"{dat0.shape} / {list(dat0.columns)}")
    show("flag 分布", dat0.flag.value_counts().to_dict())

    # 1 BFS：从已知欺诈者出发找所有关联申请人/联系人
    ids, seen = {SEED_FRAUD}, set()
    for _ in range(6):
        sub = dat0[dat0.userid.isin(ids) | dat0.connid.isin(ids)]
        new = set(sub.userid) | set(sub.connid)
        if new <= ids:
            break
        ids = ids | new
    related = dat0[dat0.userid.isin(ids) | dat0.connid.isin(ids)]
    show("已知欺诈者", SEED_FRAUD)
    show("BFS 关联到的节点数 / 关系条数", f"{len(ids)} / {len(related)}")
    show("关联样本示例", related.head(3).to_string(index=False))

    G_seed = build_graph(related)
    fig, ax = plt.subplots(figsize=(7, 6))
    pos = nx.spring_layout(G_seed, seed=7, k=0.5)
    colors = ["#ff7b7b" if G_seed.nodes[n].get("flag") == 1 else "#5aa9ff"
              for n in G_seed.nodes]
    nx.draw_networkx(G_seed, pos, node_color=colors, node_size=45,
                     with_labels=False, alpha=.8, ax=ax)
    ax.set_title(f"与 {SEED_FRAUD} 关联的子网（红=欺诈）")
    ax.axis("off")
    fig.tight_layout()
    save(fig, "欺诈者关联子网")

    # 2 全量（或限量）网络 + Louvain 社区发现
    net = dat0 if NET_EDGES is None else dat0.iloc[:NET_EDGES]
    G = build_graph(net)
    show("建图规模", f"节点 {G.number_of_nodes()} / 边 {G.number_of_edges()}")
    show("连通分量数", nx.number_connected_components(G))
    show("度最高的 5 个节点",
         sorted(G.degree, key=lambda x: -x[1])[:5])

    part = best_partition(G)
    part_s = pd.Series(part, name="group")
    show("社区数 / 最大社区规模",
         f"{part_s.nunique()} / {part_s.value_counts().max()}")
    fraud_groups = (dat0[dat0.flag == 1][["userid"]].drop_duplicates()
                    .set_index("userid").join(part_s).group.dropna().unique())
    show("欺诈者所在社区编号（前 10）", fraud_groups[:10])
    show("欺诈者所在社区规模（Top 5）",
         part_s[part_s.isin(fraud_groups)].value_counts().head(5))

    # 3 社区着色图（只画节点数适中的前 3 个社区，避免图太密）
    top_groups = part_s.value_counts().head(3).index.tolist()
    nodes = [n for n, g in part.items() if g in top_groups]
    sub = G.subgraph(nodes)
    fig, ax = plt.subplots(figsize=(8, 6))
    pos2 = nx.spring_layout(sub, seed=3, k=0.35)
    cmap = plt.get_cmap("tab10")
    color_map = [cmap(top_groups.index(part[n]) % 10) for n in sub.nodes]
    nx.draw_networkx(sub, pos2, node_color=color_map, node_size=22,
                     with_labels=False, alpha=.75, ax=ax)
    ax.set_title("Louvain 社区发现（前 3 大社区）")
    ax.axis("off")
    fig.tight_layout()
    save(fig, "社区发现")

    # 4 时序网络演化：按申请日期逐日累积建图
    dates = sorted(dat0.in_date.dropna().unique())[:40]
    rows, g_acc = [], nx.Graph()
    for d in dates:
        day = dat0[dat0.in_date == d]
        for r in day.itertuples(index=False):
            g_acc.add_edge(r.userid, r.connid)
            g_acc.nodes[r.userid]["flag"] = r.flag
            g_acc.nodes[r.connid]["flag"] = r.flag
        deg = dict(g_acc.degree())
        rows.append({"date": d, "节点数": g_acc.number_of_nodes(),
                     "边数": g_acc.number_of_edges(),
                     "平均度": np.mean(list(deg.values())) if deg else 0,
                     "欺诈节点占比": np.mean(
                         [v.get("flag") == 1 for v in g_acc.nodes.values()])})
    evo = pd.DataFrame(rows)
    show("网络演化（前 5 天 / 最后 5 天）",
         pd.concat([evo.head(), evo.tail()]).round(4).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(range(len(evo)), evo.节点数, label="节点数")
    ax[0].plot(range(len(evo)), evo.边数, label="边数")
    ax[0].set(title="网络规模随申请日累积", xlabel="日序"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(range(len(evo)), evo.欺诈节点占比, color="#ff7b7b")
    ax[1].set(title="欺诈节点占比", xlabel="日序"); ax[1].grid(alpha=.3)
    fig.tight_layout()
    save(fig, "网络时序演化")


if __name__ == "__main__" or True:
    main()
