# -*- coding: utf-8 -*-
"""
04_visuals.py —— 商业数据分析课程：全部 18 张图（前 12 张教材家族 + 后 6 张商业模型）
====================================================================================
严格按教材 PPT《Excel数据可视化——从图表到数据大屏》（郭宏远/清华）的版式复现：
  · 页面底色    NAVY  = #151C48  （从 pptx 的 slideLayout solidFill 抽得）
  · 浅色文字    LIGHT = #F1F1FB  （layout 浅色前景）
  · 强调蓝      BLUE_A= #3283FF  （layout 强调色）
  · 系列商务色  SER   = #0070C0 / #E74E69 / #FFC000 / #82ADD7 / #7030A0 / #A6A6A6 / #C55A11 / #70AD47
                  （从 pptx 内嵌图表例子的真实像素色抽得，与原 04_visuals_textbook 的 TB 一致）
  · 字体        微软雅黑（Microsoft YaHei），与 PPT theme1 的 major/minor 字体一致

防遮挡/防重合规则（你提的硬要求）：
  · 所有图例一律放在「绘图区之外」（图下方居中，或留白角），绝不压在柱/线/点上；
  · 数据标签统一加深色描边（path_effects），在深蓝柱、金黄柱、红柱上都清晰可读；
  · 标题用 bold + 足够 pad，出图用 bbox_inches="tight"，不裁标题、不裁图例；
  · 散点/气泡图用 margins 留白，点不被坐标轴切掉；热力图用自定义藏青→浅色渐变，单元注释带描边。

运行：脚本直跑 / Jupyter 粘贴均可。依赖 pandas numpy matplotlib scikit-learn statsmodels。
（本脚本自包含重算六模型，可脱离 03_analysis 独立出图；也可先跑 03 再用其 tables。）
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap

# ---------------- 路径配置（兼容脚本 / Jupyter）----------------
try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:
    REPO = Path(r"C:/Users/LENOVO/Desktop/repo")
CLEAN = REPO / "data" / "clean"
FIG = REPO / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# =================== 严格 PPT 版式常量 ===================
NAVY   = "#151C48"   # 幻灯片页面底色
LIGHT  = "#F1F1FB"   # 浅色文字 / 前景
BLUE_A = "#3283FF"   # 强调蓝
SER = ["#0070C0", "#E74E69", "#FFC000", "#82ADD7",
       "#7030A0", "#A6A6A6", "#C55A11", "#70AD47"]
BLUE, RED, AMBER, LBLUE, PURPLE, GREY, ORANGE, GREEN = SER

# ---------------- 字体（与 PPT 一致：微软雅黑）----------------
for cand in ["Microsoft YaHei", "SimHei", "Microsoft JhengHei",
             "WenQuanYi Micro Hei", "Arial Unicode MS"]:
    try:
        fm.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 全局暗色主题（贴合 PPT 藏青底）----------------
plt.rcParams.update({
    "figure.facecolor": NAVY,
    "axes.facecolor":  NAVY,
    "savefig.facecolor": NAVY,
    "text.color":      LIGHT,
    "axes.labelcolor": LIGHT,
    "axes.titlecolor": LIGHT,
    "xtick.color":     LIGHT,
    "ytick.color":     LIGHT,
    "axes.edgecolor":  LIGHT,
    "legend.frameon":  False,
    "axes.grid":       False,
})

# 自定义热力图渐变：藏青(低) → 蓝 → 浅蓝 → 浅色(高)
CMAP_RET = LinearSegmentedColormap.from_list("ppt_ret", [NAVY, BLUE, LBLUE, LIGHT])

# ---------------- 通用样式工具 ----------------
def style(ax, title=None):
    """去网格线、去上/右边框、左/下边框用浅色；标题加粗留白。"""
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LIGHT)
    ax.spines["bottom"].set_color(LIGHT)
    if title:
        ax.set_title(title, fontweight="bold", fontsize=13, pad=14, color=LIGHT)

def legible(color=LIGHT, outline=NAVY):
    """标签描边：在任意底色/柱色上都清晰，不会被遮挡看不清。"""
    return [pe.withStroke(linewidth=2.3, foreground=outline)]

def add_labels(ax, containers, fmt="%.1f", fs=7, pad=2):
    for c in containers:
        for t in ax.bar_label(c, fmt=fmt, fontsize=fs, padding=pad):
            t.set_color(LIGHT)
            t.set_path_effects(legible())

def legend_below(ax, ncol=3, title=None, fs=9, labels=None):
    """图例一律放图下方居中，绝不压数据。饼图等无自动标签的场景需传 labels。"""
    kw = {"labels": labels} if labels is not None else {}
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=ncol, frameon=False, labelcolor=LIGHT, fontsize=fs, title=title, **kw)

def save(name, dpi=150):
    fig = plt.gcf()
    fig.savefig(FIG / name, dpi=dpi, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig)
    print(f"  -> {name}")

# =================== 载入数据 & 派生 ===================
df = pd.read_csv(CLEAN / "orders_clean.csv", parse_dates=["Order Date"])
if "Order Year" not in df:  df["Order Year"]  = df["Order Date"].dt.year
if "Order Month" not in df: df["Order Month"] = df["Order Date"].dt.to_period("M").astype(str)
print(f"[读入] orders_clean {df.shape}")

monthly = df.groupby("Order Month").agg(销售额=("Sales", "sum"),
                                        利润=("Profit", "sum")).reset_index()

# =====================================================================
#  第一部分：前 12 张 —— 教材《Excel数据可视化》第二章 图表家族复现
#  （柱形/条形、组合、折线、面积、饼/环形，贴近原模板布局/配色/标签/图例）
# =====================================================================
print("\n========== 前 12 张：教材图表家族复现（严格 PPT 暗色版式）==========")

# ---- 01 单系列柱状图 ----
cat = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7, 4.6))
bars = ax.bar(cat.index, cat.values / 1e6, color=[BLUE, LBLUE, AMBER][:len(cat)], width=0.62)
ax.set_ylabel("销售额（百万）")
style(ax, "各产品类别销售额")
add_labels(ax, [bars])
save("figT01_单系列柱状图.png")

# ---- 02 多系列柱状图(2系) ----
piv2 = df.pivot_table(index="Region", columns="Segment", values="Sales", aggfunc="sum").fillna(0)
piv2 = piv2.loc[piv2.sum(1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(8.2, 4.8))
(piv2 / 1e6).plot(kind="bar", ax=ax, color=[BLUE, RED, AMBER][:piv2.shape[1]], width=0.8)
ax.set_ylabel("销售额（百万）"); ax.set_xlabel("")
style(ax, "各区域分客群销售额")
add_labels(ax, ax.containers)
legend_below(ax, ncol=piv2.shape[1], title="客群")
save("figT02_多系列柱状图2系.png")

# ---- 03 多系列柱状图(4系) ----
piv4 = df.pivot_table(index="Category", columns="Region", values="Sales", aggfunc="sum").fillna(0)
piv4 = piv4.loc[piv4.sum(1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(8.6, 4.8))
(piv4 / 1e6).plot(kind="bar", ax=ax, color=[BLUE, RED, AMBER, GREEN][:piv4.shape[1]], width=0.8)
ax.set_ylabel("销售额（百万）"); ax.set_xlabel("")
style(ax, "各类别分区域销售额")
add_labels(ax, ax.containers)
legend_below(ax, ncol=piv4.shape[1], title="区域")
save("figT03_多系列柱状图4系.png")

# ---- 04 堆叠柱状图 ----
pivs = df.pivot_table(index="Region", columns="Segment", values="Sales", aggfunc="sum").fillna(0)
pivs = pivs.loc[pivs.sum(1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(8.2, 4.8))
(pivs / 1e6).plot(kind="bar", stacked=True, ax=ax, color=[BLUE, RED, AMBER][:pivs.shape[1]], width=0.8)
ax.set_ylabel("销售额（百万）"); ax.set_xlabel("")
style(ax, "各区域客群销售额（堆叠）")
add_labels(ax, ax.containers)
legend_below(ax, ncol=pivs.shape[1], title="客群")
save("figT04_堆叠柱状图.png")

# ---- 05 组合图 柱+线 ----
fig, ax1 = plt.subplots(figsize=(9.5, 5.0))
x = np.arange(len(monthly))
ax1.bar(x, monthly["销售额"] / 1e6, color=BLUE, alpha=0.85, label="销售额")
ax1.set_ylabel("销售额（百万）", color=LIGHT)
ax1.set_xticks(x); ax1.set_xticklabels(monthly["Order Month"], rotation=90, fontsize=6)
ax1.set_xlabel("")
ax2 = ax1.twinx()
ax2.plot(x, monthly["利润"] / 1e6, color=AMBER, marker="o", ms=3, lw=1.6, label="利润")
ax2.set_ylabel("利润（百万）", color=LIGHT)
ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_color(LIGHT)
ax2.spines["right"].set_color(LIGHT)
ax2.tick_params(colors=LIGHT)
ax1.set_ylim(top=(monthly["销售额"].max() / 1e6) * 1.22)   # 顶部留白，图例不压柱
ax2.set_ylim(top=(monthly["利润"].max() / 1e6) * 1.30)
style(ax1, "月度销售额（柱）与利润（线）")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
h = [Patch(color=BLUE, label="销售额"), Line2D([0], [0], color=AMBER, marker="o", label="利润")]
ax1.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.22),
           ncol=2, frameon=False, labelcolor=LIGHT, fontsize=9)
save("figT05_组合图柱线.png")

# ---- 06 单系列折线图 ----
fig, ax = plt.subplots(figsize=(9.5, 4.4))
ax.plot(range(len(monthly)), monthly["销售额"] / 1e6, color=BLUE, marker="o", ms=3, lw=1.6)
ax.set_ylabel("销售额（百万）")
ax.set_xlabel(f"{monthly['Order Month'].iloc[0]} ~ {monthly['Order Month'].iloc[-1]}")
ax.set_xticks([])
style(ax, "月度销售额趋势")
save("figT06_单系列折线图.png")

# ---- 07 多系列折线图 ----
pivm = df.pivot_table(index="Order Month", columns="Segment", values="Sales", aggfunc="sum").fillna(0)
fig, ax = plt.subplots(figsize=(9.5, 4.4))
for i, seg in enumerate(pivm.columns):
    ax.plot(range(len(pivm)), pivm[seg].values / 1e6, marker="o", ms=2.5, lw=1.4,
            color=[BLUE, RED, AMBER][i], label=seg)
ax.set_ylabel("销售额（百万）")
ax.set_xlabel(f"{pivm.index[0]} ~ {pivm.index[-1]}")
ax.set_xticks([])
style(ax, "各客群月度销售额趋势")
legend_below(ax, ncol=len(pivm.columns), title="客群")
save("figT07_多系列折线图.png")

# ---- 08 环形图 ----
seg = df.groupby("Segment")["Sales"].sum()
fig, ax = plt.subplots(figsize=(6.4, 6.0))
wedges, _, autotexts = ax.pie(seg.values, colors=[BLUE, RED, AMBER][:len(seg)],
        autopct=lambda p: f"{p:.1f}%", pctdistance=0.78,
        wedgeprops=dict(width=0.42, edgecolor=NAVY))
for at in autotexts:
    at.set_fontsize(9); at.set_color(LIGHT); at.set_path_effects(legible())
ax.set_title("客群销售额占比", fontweight="bold", fontsize=13, pad=14, color=LIGHT)
legend_below(ax, ncol=len(seg), title="客群", labels=list(seg.index))
save("figT08_环形图.png")

# ---- 09 饼图 ----
cat2 = df.groupby("Category")["Sales"].sum()
fig, ax = plt.subplots(figsize=(6.4, 6.0))
wedges, _, autotexts = ax.pie(cat2.values, colors=[BLUE, LBLUE, AMBER][:len(cat2)],
        autopct=lambda p: f"{p:.1f}%", startangle=90,
        wedgeprops=dict(edgecolor=NAVY))
for at in autotexts:
    at.set_fontsize(9); at.set_color(LIGHT); at.set_path_effects(legible())
ax.set_title("产品类别销售额占比", fontweight="bold", fontsize=13, pad=14, color=LIGHT)
legend_below(ax, ncol=len(cat2), title="类别", labels=list(cat2.index))
save("figT09_饼图.png")

# ---- 10 多环环形图 ----
fig, ax = plt.subplots(figsize=(7.4, 7.0))
inner = df.groupby("Segment")["Sales"].sum().sort_values(ascending=False)
outer = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
ax.pie(inner.values, radius=1.0, colors=[BLUE, RED, AMBER][:len(inner)],
       wedgeprops=dict(width=0.35, edgecolor=NAVY),
       autopct=lambda p: f"{p:.1f}%", pctdistance=0.82,
       textprops=dict(color=LIGHT, fontsize=8))
ax.pie(outer.values, radius=0.62, colors=[LBLUE, GREEN, PURPLE][:len(outer)],
       wedgeprops=dict(width=0.35, edgecolor=NAVY),
       autopct=lambda p: f"{p:.1f}%", pctdistance=0.82,
       textprops=dict(color=LIGHT, fontsize=7))
for t in ax.texts:
    t.set_path_effects(legible())
ax.text(0, 0, "内层:客群\n外层:类别", ha="center", va="center",
        fontsize=9, color=LIGHT, path_effects=legible())
ax.set_title("类别与客群销售额（双环）", fontweight="bold", fontsize=13, pad=14, color=LIGHT)
save("figT10_多环环形图.png")

# ---- 11 面积图 ----
fig, ax = plt.subplots(figsize=(9.5, 4.4))
cum = monthly["销售额"].cumsum().values / 1e6
ax.fill_between(range(len(monthly)), cum, color=BLUE, alpha=0.30)
ax.plot(range(len(monthly)), cum, color=BLUE, lw=1.6)
ax.set_ylabel("累计销售额（百万）")
ax.set_xlabel(f"{monthly['Order Month'].iloc[0]} ~ {monthly['Order Month'].iloc[-1]}")
ax.set_xticks([])
style(ax, "累计销售额")
save("figT11_面积图.png")

# ---- 12 横向条形图 ----
sub = df.groupby("Sub-Category")["Profit"].sum().sort_values()
fig, ax = plt.subplots(figsize=(8.2, 6.4))
colors = [GREEN if v >= 0 else RED for v in sub.values]
ax.barh(sub.index, sub.values / 1e3, color=colors)
ax.axvline(0, color=LIGHT, lw=0.9)
ax.set_xlabel("利润（千元）")
style(ax, "各子类别利润")
for i, v in enumerate(sub.values / 1e3):
    ax.annotate(f"{v:.0f}", (v, i), xytext=(4 if v >= 0 else -4, 0),
                textcoords="offset points", va="center",
                ha="left" if v >= 0 else "right", fontsize=6,
                color=LIGHT, path_effects=legible())
save("figT12_横向条形图.png")

# =====================================================================
#  第二部分：后 6 张 —— 商业分析六模型（RFM / K-Means / ABC / 波士顿 / 同期群 / SARIMA）
#  数据与切分逻辑与 03_analysis.py 完全一致，此处自包含重算，可独立出图。
# =====================================================================
print("\n========== 后 6 张：商业分析六模型（严格 PPT 暗色版式）==========")

# ---------- 1. RFM 八分类 ----------
snapshot = df["Order Date"].max()
rfm = df.groupby("Customer ID").agg(
    R=("Order Date", lambda s: (snapshot - s.max()).days),
    F=("Order ID", "nunique"),
    M=("Sales", "sum"),
).reset_index()
r_med, f_med, m_med = rfm["R"].median(), rfm["F"].median(), rfm["M"].median()
rfm["R_s"] = (rfm["R"] <= r_med).astype(int)
rfm["F_s"] = (rfm["F"] >= f_med).astype(int)
rfm["M_s"] = (rfm["M"] >= m_med).astype(int)
seg_keys = {(1,1,1):"重要价值",(1,1,0):"重要保持",(1,0,1):"重要发展",(1,0,0):"重要挽留",
            (0,1,1):"一般价值",(0,1,0):"一般保持",(0,0,1):"一般发展",(0,0,0):"一般挽留"}
rfm["Segment"] = rfm.apply(lambda r: seg_keys[(r["R_s"], r["F_s"], r["M_s"])], axis=1)
seg_tab = rfm.groupby("Segment").agg(人数=("Customer ID","count"), 销售额=("M","sum")).reset_index()
seg_tab["销售额占比"] = (seg_tab["销售额"]/seg_tab["销售额"].sum()*100).round(1)
seg_tab = seg_tab.sort_values("销售额", ascending=False)

fig, ax = plt.subplots(figsize=(9.0, 5.2))
cmap8 = [BLUE, RED, AMBER, LBLUE, GREEN, PURPLE, ORANGE, GREY]
bars = ax.bar(seg_tab["Segment"], seg_tab["销售额"]/1e6, color=cmap8, width=0.66)
ax.set_ylabel("销售额（百万）")
style(ax, "RFM 八类客户销售额分布")
add_labels(ax, [bars], fmt="%.2f")
ax.set_xticklabels(seg_tab["Segment"], rotation=30, ha="right", fontsize=8)
save("fig13_RFM八类客户.png")

# ---------- 2. K-Means 客群聚类 ----------
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
X = StandardScaler().fit_transform(rfm[["R","F","M"]])
inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_ for k in range(1, 9)]
K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
rfm["Cluster"] = km.labels_

fig, ax = plt.subplots(figsize=(8.6, 5.4))
cc = [BLUE, RED, AMBER, GREEN]
for c in range(K):
    m = rfm["Cluster"] == c
    ax.scatter(rfm.loc[m, "M"], rfm.loc[m, "F"], s=9, alpha=0.55,
               color=cc[c], label=f"簇{c+1}", edgecolors="none")
ax.set_xlabel("销售额 M（元）"); ax.set_ylabel("购买频次 F（次）")
style(ax, "K-Means 客群聚类（K=4，M–F 平面）")
ax.legend(loc="upper right", frameon=False, labelcolor=LIGHT, fontsize=9,
          title="客群簇", title_fontsize=9)
ax.margins(0.08)
save("fig14_KMeans客群聚类.png")

# ---------- 3. ABC 品类分级（帕累托）----------
abc = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).reset_index()
abc["累计占比"] = abc["Sales"].cumsum()/abc["Sales"].sum()*100
def grade(p):
    return "A" if p < 80 else ("B" if p < 95 else "C")
abc["等级"] = abc["累计占比"].apply(grade)
gcol = {"A": BLUE, "B": AMBER, "C": GREY}

fig, ax1 = plt.subplots(figsize=(10.5, 5.4))
bars = ax1.bar(abc["Sub-Category"], abc["Sales"]/1e6,
               color=[gcol[g] for g in abc["等级"]], width=0.82)
ax1.set_ylabel("销售额（百万）"); ax1.set_xlabel("")
ax1.set_xticklabels(abc["Sub-Category"], rotation=90, fontsize=6)
ax2 = ax1.twinx()
ax2.plot(range(len(abc)), abc["累计占比"], color=RED, marker="o", ms=3, lw=1.5)
ax2.axhline(80, color=LIGHT, ls="--", lw=0.9); ax2.axhline(95, color=LIGHT, ls=":", lw=0.9)
ax2.set_ylim(0, 105); ax2.set_ylabel("累计占比（%）")
ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_color(LIGHT); ax2.spines["right"].set_color(LIGHT)
ax2.tick_params(colors=LIGHT)
style(ax1, "ABC 品类帕累托分析")
from matplotlib.lines import Line2D as _L
h = [Patch(color=BLUE, label="A 类"), Patch(color=AMBER, label="B 类"),
     Patch(color=GREY, label="C 类"), _L([0],[0], color=RED, marker="o", label="累计占比")]
ax1.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.20),
           ncol=4, frameon=False, labelcolor=LIGHT, fontsize=9)
save("fig15_ABC帕累托.png")

# ---------- 4. 波士顿矩阵 ----------
year_sales = df.pivot_table(index="Sub-Category", columns="Order Year", values="Sales", aggfunc="sum")
last, prev = year_sales.columns.max(), year_sales.columns.max() - 1
bos = pd.DataFrame({"份额": df.groupby("Sub-Category")["Sales"].sum()/df["Sales"].sum()*100})
bos["增速"] = (year_sales[last] - year_sales[prev]) / year_sales[prev] * 100
bos["利润"] = df.groupby("Sub-Category")["Profit"].sum()
bos["增速"] = bos["增速"].fillna(-1)
s_med, g_med = bos["份额"].median(), bos["增速"].median()
qkeys = {(1,1):"明星",(1,0):"金牛",(0,1):"问题",(0,0):"瘦狗"}
bos["象限"] = bos.apply(lambda r: qkeys[(r["份额"]>=s_med, r["增速"]>=g_med)], axis=1)
qcol = {"明星": BLUE, "金牛": GREEN, "问题": AMBER, "瘦狗": GREY}

fig, ax = plt.subplots(figsize=(9.0, 6.2))
for q in ["明星","金牛","问题","瘦狗"]:
    m = bos["象限"] == q
    ax.scatter(bos.loc[m,"份额"], bos.loc[m,"增速"],
               s=(bos.loc[m,"利润"].abs()/bos["利润"].abs().max()*420 + 30),
               color=qcol[q], alpha=0.75, label=q, edgecolors=LIGHT, linewidths=0.6)
ax.axvline(s_med, color=LIGHT, ls="--", lw=0.9)
ax.axhline(g_med, color=LIGHT, ls="--", lw=0.9)
ax.set_xlabel("销售额份额（%）"); ax.set_ylabel("同比增速（%）")
style(ax, "波士顿矩阵（气泡大小=利润）")
for name, row in bos.iterrows():
    ax.annotate(name, (row["份额"], row["增速"]), xytext=(4, 4),
                textcoords="offset points", fontsize=6, color=LIGHT, path_effects=legible())
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
          ncol=4, frameon=False, labelcolor=LIGHT, fontsize=9, title="象限")
ax.margins(0.12)
save("fig16_波士顿矩阵.png")

# ---------- 5. 同期群留存热力图 ----------
df["mo_ord"] = df["Order Date"].dt.to_period("M").apply(lambda p: p.year*12 + p.month)
first_ord = df.groupby("Customer ID")["mo_ord"].transform("min")
df["首购月"] = pd.PeriodIndex(df.groupby("Customer ID")["Order Date"].transform("min"), freq="M")
df["期数"] = df["mo_ord"] - first_ord
coh = df.groupby(["首购月","期数"])["Customer ID"].nunique().reset_index()
piv = coh.pivot(index="首购月", columns="期数", values="Customer ID")
size = piv[0]
ret = piv.div(size, axis=0) * 100
n_per = ret.shape[1]
ret = ret.apply(lambda row: row.where(row.index <= (n_per - 1 - ret.index.get_loc(row.name)),
                                      other=np.nan), axis=1)
ret = ret.round(1)
Z = ret.values.astype(float)
ylabels = [str(p) for p in ret.index]
xlabels = [str(c) for c in ret.columns]

fig, ax = plt.subplots(figsize=(11.0, 7.2))
im = ax.imshow(Z, cmap=CMAP_RET, aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, fontsize=7)
ax.set_yticks(range(len(ylabels))); ax.set_yticklabels(ylabels, fontsize=7)
ax.tick_params(axis="y", pad=8)
ax.set_xlim(-0.7, len(xlabels) - 0.3)
ax.set_xlabel("期数（自首购起第 N 个月）"); ax.set_ylabel("首购月同期群")
style(ax, "同期群留存率热力图（%）")
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        v = Z[i, j]
        if not np.isnan(v):
            if v > 55:   # 亮格：深字 + 浅描边
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                        color=NAVY, path_effects=[pe.withStroke(linewidth=1.8, foreground=LIGHT)])
            else:        # 暗格：浅字 + 深描边
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                        color=LIGHT, path_effects=legible())
cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.035)
cb.set_label("留存率（%）", color=LIGHT); cb.ax.yaxis.set_tick_params(color=LIGHT)
cb.outline.set_edgecolor(LIGHT)
save("fig17_同期群留存热力图.png")

# ---------- 6. SARIMA 销售额预测 ----------
from statsmodels.tsa.statespace.sarimax import SARIMAX
ms = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum().sort_index()
ts = ms.to_timestamp().asfreq("MS")
model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
train = ts.iloc[:-12]; test = ts.iloc[-12:]
m2 = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
fc = m2.get_forecast(steps=12)
pred = fc.predicted_mean
fc_all = model.get_forecast(steps=12)
mean_f = fc_all.predicted_mean
ci = fc_all.conf_int(alpha=0.05)

fig, ax = plt.subplots(figsize=(11.0, 5.0))
ax.plot(range(len(ts)), ts.values/1e6, color=BLUE, lw=1.6, label="历史实际")
fx = range(len(ts), len(ts) + len(mean_f))
ax.plot(fx, mean_f.values/1e6, color=AMBER, lw=1.8, marker="o", ms=3, label="预测")
ax.fill_between(fx, ci.iloc[:,0].values/1e6, ci.iloc[:,1].values/1e6,
                color=LBLUE, alpha=0.25, label="95% 置信区间")
ax.set_ylabel("销售额（百万）")
ax.set_xlabel(f"{ts.index[0].strftime('%Y-%m')} ~ {mean_f.index[-1].strftime('%Y-%m')}")
ax.set_xticks([])
style(ax, "SARIMA 月度销售额预测（12 个月）")
legend_below(ax, ncol=3, title="序列")
save("fig18_SARIMA预测.png")

print("\n[完成] 全部 18 张图已写入 outputs/figures/（figT01–figT12 + fig13–fig18）")
print("版式：藏青底 #151C48 / 浅字 #F1F1FB / 系列商务色 / 微软雅黑；图例与标签均避让数据、不遮挡。")
