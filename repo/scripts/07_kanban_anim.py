# -*- coding: utf-8 -*-
"""
07_kanban_anim.py —— 生成"数据大屏"动画 GIF（会动的图片）
动作: 饼图环形扫出 + KPI 数字滚动 + 排名柱状图生长 + 面积图逐步描线
输出: outputs/figures/fig20_数据大屏动图.gif
可直接粘进 Jupyter 单个 cell 运行（含 __file__ 兜底）。
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Wedge

# ---------- 路径 ----------
try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:
    REPO = Path(r"C:/Users/LENOVO/Desktop/repo")
CSV = REPO / "data" / "clean" / "orders_clean.csv"
FIGDIR = REPO / "outputs" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---------- 配色（教材第四章看板） ----------
BG     = "#10162E"
PANEL  = "#2C3560"
LIGHT  = "#F1F1FB"
ACCENT = "#3283FF"
SERIES = ["#3283FF", "#E74E69", "#FFC000", "#82ADD7", "#7030A0",
          "#70AD47", "#C55A11", "#A6A6A6"]

# ---------- 中文字体 ----------
for f in ["Microsoft YaHei", "SimHei", "DengXian", "Arial Unicode MS"]:
    wf = os.path.join(os.environ.get("WINDIR", ""), "Fonts")
    if os.path.isdir(wf) and any(f.lower() in x.lower() for x in os.listdir(wf)):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------- 数据 ----------
df = pd.read_csv(CSV)
df["Order Date"] = pd.to_datetime(df["Order Date"])
sales = df["Sales"].sum()
profit = df["Profit"].sum()
margin = profit / sales * 100
orders = df["Order ID"].nunique()
custs = df["Customer ID"].nunique()
aov = df.groupby("Order ID")["Sales"].sum().mean()

# ---------- 画布 ----------
FW, FH, DPI = 16.0, 9.0, 120
fig = plt.figure(figsize=(FW, FH), dpi=DPI)
fig.patch.set_facecolor(BG)

def panel(x, y, w, h, title):
    fig.add_artist(FancyBboxPatch((x, y), w, h,
                   boxstyle="round,pad=0.004,rounding_size=0.012",
                   transform=fig.transFigure, facecolor=PANEL,
                   edgecolor="none", zorder=0))
    if title:
        fig.text(x + 0.012, y + h - 0.028, title, color=LIGHT,
                 fontsize=11.5, fontweight="bold", va="top", zorder=3)

panel(0, 0.925, 1, 0.075, "")
fig.text(0.5, 0.962, "公 司 销 售 数 据 可 视 化 大 屏", color=LIGHT,
         fontsize=21, fontweight="bold", ha="center", va="center")
fig.text(0.978, 0.962, "数据来源：Superstore 零售交易数据", color="#9AA3C7",
         fontsize=9.5, ha="right", va="center")

# KPI 卡
def fmt_wan(v):
    return f"{v/10000:,.1f} 万" if v >= 10000 else f"{v:,.0f}"
kpis = [("总销售额", sales, "wan"), ("总利润", profit, "wan"),
        ("利润率", margin, "pct"), ("订单数", orders, "int"),
        ("客户数", custs, "int"), ("平均客单价", aov, "wan")]
kw, gap, x0, ky, kh = 0.1505, 0.0098, 0.020, 0.795, 0.105
kpi_texts = []
for i, (name, val, kind) in enumerate(kpis):
    x = x0 + i * (kw + gap)
    panel(x, ky, kw, kh, "")
    t = fig.text(x + kw/2, ky + kh*0.68, "", color=ACCENT if i in (0, 1) else LIGHT,
                 fontsize=16.5, fontweight="bold", ha="center", va="center", zorder=3)
    fig.text(x + kw/2, ky + kh*0.25, name, color="#9AA3C7",
             fontsize=10.5, ha="center", va="center")
    kpi_texts.append((t, val, kind))

# 面积图
panel(0.020, 0.425, 0.545, 0.345, "月度销售额趋势（面积图）")
m = (df.assign(ym=df["Order Date"].dt.to_period("M"))
       .groupby("ym")["Sales"].sum().sort_index())
mx = np.arange(len(m))
ax1 = fig.add_axes([0.055, 0.46, 0.49, 0.26]); ax1.patch.set_alpha(0)
ax1.set_xlim(0, len(m) - 1); ax1.set_ylim(0, m.values.max() * 1.15)
ax1.set_xticks(mx[::max(1, len(m)//10)])
ax1.set_xticklabels([str(p) for p in m.index[::max(1, len(m)//10)]],
                    fontsize=7.5, color="#9AA3C7")
ax1.tick_params(colors="#9AA3C7", labelsize=8); ax1.set_yticks([])
for s in ax1.spines.values(): s.set_visible(False)
ax1.grid(axis="y", color="#3A4470", lw=0.5, alpha=0.5); ax1.set_axisbelow(True)
fill_coll = ax1.fill_between(mx, 0, 0, color=ACCENT, alpha=0.35, zorder=1)
line1, = ax1.plot([], [], color=ACCENT, lw=2, zorder=2)

# Top10 横条
panel(0.585, 0.425, 0.395, 0.345, "子类别销售额 Top 10")
top10 = df.groupby("Sub-Category")["Sales"].sum().sort_values().tail(10)
ax2 = fig.add_axes([0.665, 0.455, 0.295, 0.275]); ax2.patch.set_alpha(0)
ax2.set_xlim(0, top10.values.max() * 1.22); ax2.set_ylim(-0.5, 9.5)
ax2.set_yticks(range(10)); ax2.set_yticklabels(top10.index, color=LIGHT, fontsize=9)
ax2.tick_params(axis="y", length=0); ax2.set_xticks([])
for s in ax2.spines.values(): s.set_visible(False)
bars = ax2.barh(range(10), np.zeros(10), color=ACCENT, height=0.62)
bar_labels = [ax2.text(0, i, "", va="center", color=LIGHT, fontsize=8.5)
              for i in range(10)]

# 4 个环形
donut_defs = [
    ("区域销售额占比",    df.groupby("Region")["Sales"].sum().sort_values(ascending=False)),
    ("客群销售额占比",    df.groupby("Segment")["Sales"].sum().sort_values(ascending=False)),
    ("品类销售额占比",    df.groupby("Category")["Sales"].sum().sort_values(ascending=False)),
    ("运输方式销售额占比", df.groupby("Ship Mode")["Sales"].sum().sort_values(ascending=False)),
]
donut_axes = []
for name, vc in donut_defs:
    x = [0.020, 0.268, 0.516, 0.764][len(donut_axes)]
    panel(x, 0.035, 0.216, 0.355, name)
    axd = fig.add_axes([x + 0.045, 0.115, 0.13, 0.20]); axd.patch.set_alpha(0)
    axd.set_aspect("equal"); axd.axis("off")
    axd.set_xlim(-1.3, 1.3); axd.set_ylim(-1.3, 1.3)
    axd.text(0, 0.08, "", color=LIGHT, ha="center", fontsize=12.5, fontweight="bold")
    axd.text(0, -0.22, "合计", color="#9AA3C7", ha="center", fontsize=8.5)
    cols = [SERIES[i % len(SERIES)] for i in range(len(vc))]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
    axd.legend(handles, vc.index, loc="upper center", bbox_to_anchor=(0.5, 0.02),
               ncol=2, frameon=False, fontsize=8.5, labelcolor=LIGHT,
               handlelength=1.0, columnspacing=0.9)
    donut_axes.append((axd, vc, cols, []))

# ---------- 动画（逐帧渲染后拼 GIF，规避 PillowWriter 空帧 bug） ----------
N = 50
DPI = 100
def update_state(p):
    global fill_coll
    ease = 1 - (1 - p) ** 2            # 缓出
    for t, val, kind in kpi_texts:
        v = val * ease
        t.set_text(fmt_wan(v) if kind == "wan" else
                   (f"{v:.1f}%" if kind == "pct" else f"{int(round(v)):,}"))
    k = max(1, int(ease * len(m)))
    line1.set_data(mx[:k], m.values[:k])
    fill_coll.remove()
    fill_coll = ax1.fill_between(mx[:k], 0, m.values[:k], color=ACCENT, alpha=0.35, zorder=1)
    for i, b in enumerate(bars):
        b.set_width(top10.values[i] * ease)
        bar_labels[i].set_x(top10.values[i] * ease)
        bar_labels[i].set_text(f"  {top10.values[i]*ease/10000:.1f}万")
    for axd, vc, cols, wedges in donut_axes:
        for w in wedges:
            w.remove()
        wedges.clear()
        total = vc.sum(); cum = 0
        for val, c in zip(vc.values, cols):
            frac = val / total
            w = Wedge((0, 0), 1.0, cum*360, (cum + frac*ease)*360,
                      width=0.42, facecolor=c, edgecolor=PANEL, linewidth=1.5)
            axd.add_patch(w); wedges.append(w); cum += frac
        axd.texts[0].set_text(fmt_wan(total * ease))

import io
from PIL import Image
out = FIGDIR / "fig20_数据大屏动图.gif"
frames = []
for i in range(N):
    update_state(i / (N - 1))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, facecolor=BG)
    buf.seek(0)
    frames.append(Image.open(buf).convert("RGB"))
frames += [frames[-1]] * 8              # 末帧停留，便于看清终态
frames[0].save(out, save_all=True, append_images=frames[1:],
               duration=int(1000 / 12), loop=0, optimize=True)
plt.close(fig)
print("已生成:", out, "| 帧数", len(frames))
