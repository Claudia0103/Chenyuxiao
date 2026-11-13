# -*- coding: utf-8 -*-
"""
补充：社交网络（上）—— 合并 02 / 03-1 / 03-2 / 03-3 四个 notebook
====================================================================
  02 基本理论：无向图 / 有向图、加权边、度 / 出度 / 入度；
  03-1 手写邻接表 + 深度遍历(DFS) / 广度遍历(BFS)；
  03-2 四种中心度：度、紧密、介度、特征向量中心度；
  03-3 PageRank：从链接矩阵出发幂迭代，并与 networkx.pagerank 对照。

修正：教材 `MV_computer` 里循环没有把结果写回 v，迭代 1000 次其实等于只算一次；
这里改成真正的幂迭代并记录收敛曲线。中心度对不收敛的图加了 try。

数据：本专题主要用代码里构造的小图，不依赖教材 CSV。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
CHAPTER_DIR = ""
_EXPECTED = ("stormofswords.csv",)


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
    p = os.path.join(CHARTS, f"补1_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


class AdjGraph:
    """教材 03-1 的邻接表图，补上 dfs 与 bfs。"""

    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, s, t):
        self.adj[s].append(t)
        self.adj[t].append(s)

    def dfs(self, s):
        seen, order = set(), []

        def rec(v):
            seen.add(v)
            order.append(v)
            for w in self.adj[v]:
                if w not in seen:
                    rec(w)
        rec(s)
        return order

    def bfs(self, s):
        seen, order, q = {s}, [], deque([s])
        while q:
            v = q.popleft()
            order.append(v)
            for w in self.adj[v]:
                if w not in seen:
                    seen.add(w)
                    q.append(w)
        return order


def pagerank_power(M, T=300, eps=1e-8, d=0.85):
    """手写 PageRank：R_{t+1} = d M R_t + (1-d)/N，带阻尼项避免悬挂节点。"""
    N = M.shape[0]
    R = np.ones(N) / N
    hist = []
    for _ in range(T):
        R_new = d * M.dot(R) + (1 - d) / N
        hist.append(np.linalg.norm(R_new - R))
        if np.linalg.norm(R_new - R) < eps:
            R = R_new
            break
        R = R_new
    return R, hist


def main():
    nodes = [str(i) for i in range(9)]
    edges = [("0", "0", 1), ("0", "1", 1), ("0", "5", 1), ("1", "2", 3),
             ("1", "4", 5), ("2", "4", 6), ("3", "5", 5), ("4", "8", 4),
             ("6", "7", 1), ("7", "8", 2), ("8", "6", 3), ("8", "3", 5)]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    show("无向图 节点/边/度", f"{G.number_of_nodes()} / {G.number_of_edges()} / "
         f"{dict(G.degree())}")
    D = nx.DiGraph()
    D.add_nodes_from(nodes)
    D.add_weighted_edges_from(edges)
    show("有向图 节点 1 的出度 / 入度", f"{D.out_degree('1')} / {D.in_degree('1')}")

    # 手写 DFS / BFS
    g = AdjGraph(6)
    for a, b in [(0, 1), (0, 3), (1, 2), (1, 4), (3, 4), (4, 5), (2, 5)]:
        g.add_edge(a, b)
    show("手写 DFS(0) / BFS(0)", f"{g.dfs(0)} / {g.bfs(0)}")
    show("networkx BFS 树边", list(nx.bfs_edges(nx.Graph(
        {i: {j: {} for j in js} for i, js in enumerate(g.adj)}), 0))[:8])

    # 四种中心度（教材 03-2 的星形图）
    S = nx.Graph()
    S.add_nodes_from(list("ABCDEF"))
    S.add_edges_from([("A", b) for b in "BCDEF"])
    cent = pd.DataFrame({
        "度中心度": pd.Series(nx.degree_centrality(S)),
        "紧密中心度": pd.Series(nx.closeness_centrality(S)),
        "介度中心度": pd.Series(nx.betweenness_centrality(S)),
    })
    try:
        cent["特征向量中心度"] = pd.Series(nx.eigenvector_centrality(S))
    except Exception as e:
        print("  特征向量中心度跳过：", type(e).__name__)
    show("星形图四种中心度", cent.round(4))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    pos = nx.spring_layout(G, seed=7)
    nx.draw_networkx(G, pos, node_color="#5aa9ff", with_labels=True,
                     font_color="white", ax=ax[0])
    ax[0].set_title("教材 02 的无向加权图"); ax[0].axis("off")
    cent.plot(kind="bar", ax=ax[1], color=["#5aa9ff", "#8b7cff", "#3ddc97", "#ffc75a"])
    ax[1].set(title="四种中心度对比", ylabel="中心度")
    ax[1].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    save(fig, "图与中心度")

    # PageRank：教材 03-3 的 4×4 转移矩阵
    M = np.array([[0, .5, 0, .5],
                  [1/3, 0, 0, 1/3],
                  [1/3, .5, 0, 0],
                  [1/3, 0, 1, 0]])
    R, hist = pagerank_power(M)
    show("手写 PageRank 结果", np.round(R, 4))
    show("幂迭代收敛步数 / 最终变化量", f"{len(hist)} / {hist[-1]:.2e}")
    Gp = nx.DiGraph()
    for j in range(4):
        for i in range(4):
            if M[i, j] > 0:
                Gp.add_edge(j, i, weight=M[i, j])
    show("networkx.pagerank 对照",
         pd.Series(nx.pagerank(Gp, alpha=.85)).sort_index().round(4))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist, color="#8b7cff")
    ax.set(title="PageRank 幂迭代收敛", xlabel="迭代次数",
           ylabel="‖R_new − R‖", yscale="log")
    ax.grid(alpha=.3)
    fig.tight_layout()
    save(fig, "PageRank收敛")


if __name__ == "__main__" or True:
    main()
