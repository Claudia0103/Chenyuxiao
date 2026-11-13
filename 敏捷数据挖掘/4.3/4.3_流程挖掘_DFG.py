# -*- coding: utf-8 -*-
"""
4.3 流程挖掘 —— 由《流程挖掘pm4py.ipynb》整理成 .py
====================================================
教材流程：读 IEEE XES 事件日志 → 看 trace / event 结构 → 导出 XES
→ 发现 Directly-Follows Graph (DFG) → 频率视图与性能视图。

修正与替代：
  - 教材用 `pm4py.objects.log.importer.xes.importer.apply` 老接口，这里改用
    pm4py 2.7 的顶层函数 read_xes / write_xes / discover_dfg；
  - 教材的 dfg_visualization 依赖 graphviz 的 dot 二进制，本环境没装 →
    改用 networkx + matplotlib 画同一张 DFG，边标签就是频次 / 平均耗时；
  - 导出文件写到 charts/，不覆盖教材自带的 exportedLog.xes。

注意：pm4py 是 AGPL v3 许可，这里只用于本地课程复现。

数据：data/running-example.xes。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import pm4py

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('running-example.xes',)   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\敏捷数据挖掘复现\4.3流程挖掘"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '4.3流程挖掘'),
              os.path.join(os.getcwd(), '敏捷数据挖掘复现', '4.3流程挖掘')]
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
    p = os.path.join(CHARTS, f"4.3_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def draw_dfg(dfg, start, end, name, title, float_fmt=False):
    G = nx.DiGraph()
    for (a, b), w in dfg.items():
        G.add_edge(a, b, weight=float(w))
    pos = nx.spring_layout(G, seed=7, k=1.6)
    fig, ax = plt.subplots(figsize=(9, 6))
    if len(G):
        weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
        norm = (weights - weights.min()) / max(weights.max() - weights.min(), 1e-9)
        nx.draw_networkx_nodes(G, pos, node_size=1500, node_color="#5aa9ff",
                               edgecolors="#1b4b7a", ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family="SimHei", ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=16,
                               width=1 + 2.5 * norm, edge_color="#8b7cff",
                               alpha=.8, connectionstyle="arc3,rad=0.08")
        fmt = "{:.1f}" if float_fmt else "{:.0f}"
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, font_size=7,
            edge_labels={(u, v): fmt.format(G[u][v]["weight"])
                         for u, v in G.edges()})
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    save(fig, name)


def main():
    path = os.path.join(DATA, "running-example.xes")
    # pm4py 2.7 默认返回 DataFrame；教材的 log[0]/log[0][0] 是老式 EventLog 结构，
    # 用 return_legacy_log_object=True 拿回同一对象
    log = pm4py.read_xes(path, return_legacy_log_object=True)
    show("日志类型 / trace 数", f"{type(log).__name__} / {len(log)}")
    show("第一条 trace", str(log[0])[:400])
    show("第一条 trace 的第一个事件", dict(log[0][0]) if log else "空")
    show("第一条 trace 长度", len(log[0]))

    variants = pm4py.get_variants(log)
    show("流程变体数 / 最常见变体（前 3）",
         f"{len(variants)} / " + str(sorted(variants.items(),
                                          key=lambda x: -len(x[1]))[:3])[:300])
    acts = collections.Counter(ev["concept:name"] for tr in log for ev in tr)
    show("活动频次", pd.Series(acts).sort_values(ascending=False))

    # 导出 XES（等价教材的 xes_exporter.apply）
    out = os.path.join(CHARTS, "exportedLog_复现.xes")
    pm4py.write_xes(log, out)
    print(f"  导出日志：{os.path.basename(out)}")

    # 频率 DFG
    dfg, start, end = pm4py.discover_dfg(log)
    show("起始 / 结束活动", f"{dict(start)} / {dict(end)}")
    show("直接跟随关系（频次 Top 10）",
         pd.Series({f"{a} → {b}": w for (a, b), w in dfg.items()})
         .sort_values(ascending=False).head(10))
    draw_dfg(dfg, start, end, "DFG_频率视图", "DFG 频率视图（边标签=发生次数）")

    # 性能 DFG（平均耗时，单位秒）
    try:
        raw_pdfg, pstart, pend = pm4py.discover_performance_dfg(log)
        # pm4py 2.7 每条边返回 {'mean','median',...} 字典，取均值并换算成小时
        pdfg = {k: (v["mean"] if isinstance(v, dict) else float(v)) / 3600
                for k, v in raw_pdfg.items()}
        vals = {f"{a} → {b}": round(w, 2) for (a, b), w in pdfg.items()}
        show("直接跟随关系（平均耗时 Top 10，小时）",
             pd.Series(vals).sort_values(ascending=False).head(10))
        draw_dfg(pdfg, pstart, pend, "DFG_性能视图",
                 "DFG 性能视图（边标签=平均耗时/小时）", float_fmt=True)
    except Exception as e:
        print(f"性能 DFG 跳过：{type(e).__name__}: {e}")


if __name__ == "__main__" or True:
    main()
