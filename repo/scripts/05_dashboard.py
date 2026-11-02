# -*- coding: utf-8 -*-
"""
05 销售看板（Streamlit）—— 仿教材第四章「销售看板」暗色看板风格
深藏青面板 + 微软雅黑加粗标题 + KPI 卡 + 多环形占比 + 条形排名 + 面积趋势 + 地图 + 明细表。
侧边栏 Region / Segment / 年份 联动筛选，所有图表随筛选实时重算（不依赖先跑 03）。

用法（仓库根目录）：
    streamlit run scripts/05_dashboard.py
依赖：streamlit pandas plotly（见 requirements.txt）
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- 路径兜底（Jupyter / 终端 / streamlit 都能找到仓库） ----------
try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:
    REPO = Path(r"C:/Users/LENOVO/Desktop/repo")
CLEAN = REPO / "data" / "clean" / "orders_clean.csv"

# ---------- 看板配色（取自教材第四章销售看板：深藏青面板 + 浅字） ----------
PAGE_BG = "#10162E"      # 页面更深的藏青底
PANEL = "#2C3560"        # 面板填充（与参考看板一致）
PANEL2 = "#222A4E"       # 次级面板
LIGHT = "#EAEDF7"        # 浅色文字
SUB = "#9AA6D4"          # 副文字
ACCENT = "#4F8DFD"       # 强调蓝
# 环形/分类配色族
DC = ["#4F8DFD", "#E74E69", "#FFC000", "#3FC08A",
      "#9B6CF2", "#FF8C42", "#46C8D8", "#F25C9A"]
FONT = '"Microsoft YaHei", "微软雅黑", "SimHei", sans-serif'

# 州全称 -> 两字母缩写（仅美国，覆盖 50 州 + DC + PR）
STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN",
    "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "Puerto Rico": "PR",
}
RFM_NAMES = [
    "重要价值客户", "重要保持客户", "重要发展客户", "重要挽留客户",
    "一般价值客户", "一般保持客户", "一般发展客户", "一般挽留客户",
]


# ---------- 页面样式 ----------
st.set_page_config(page_title="零售销售看板", page_icon="📊", layout="wide")
st.markdown(f"""
<style>
html, body, .stApp {{ background-color: {PAGE_BG}; color: {LIGHT}; font-family: {FONT}; }}
[data-testid="stSidebar"] {{ background-color: {PANEL}; }}
h1, h2, h3 {{ color: {LIGHT} !important; font-family: {FONT}; }}
.stMarkdown {{ color: {LIGHT}; }}
.title-bar {{
    background: {PANEL}; border-radius: 10px; padding: 14px 22px; margin-bottom: 14px;
    border-left: 6px solid {ACCENT};
}}
.title-bar h1 {{ margin: 0; font-size: 22px; font-weight: 700; }}
.title-bar .sub {{ color: {SUB}; font-size: 13px; margin-top: 4px; }}
.kpi {{
    background: {PANEL}; border: 1px solid {PANEL2}; border-left: 4px solid {ACCENT};
    border-radius: 10px; padding: 14px 16px; text-align: center;
}}
.kpi-v {{ font-size: 26px; font-weight: 700; color: {LIGHT}; }}
.kpi-l {{ font-size: 13px; color: {SUB}; margin-top: 6px; }}
.kpi-s {{ font-size: 11px; color: {ACCENT}; margin-top: 2px; }}
.panel {{
    background: {PANEL}; border: 1px solid {PANEL2}; border-radius: 10px;
    padding: 6px 8px;
}}
.block-title {{
    font-size: 15px; font-weight: 700; color: {LIGHT};
    border-left: 4px solid {ACCENT}; padding-left: 10px; margin: 16px 0 8px;
}}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv(CLEAN)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["ym"] = df["Order Date"].dt.to_period("M").astype(str)
    return df


def dark(fig, h=320, legend=True):
    """统一暗色图表样式：透明底、浅字、商务色。"""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=LIGHT, size=12),
        margin=dict(l=40, r=20, t=34, b=30), height=h,
        title=dict(font=dict(size=13, color=LIGHT)),
    )
    if legend:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom",
                                     y=-0.22, xanchor="center", x=0.5,
                                     font=dict(color=LIGHT)))
    else:
        fig.update_layout(showlegend=False)
    fig.update_xaxes(gridcolor="#33406E", zerolinecolor="#33406E")
    fig.update_yaxes(gridcolor="#33406E", zerolinecolor="#33406E")
    return fig


def fmt_money(x):
    if abs(x) >= 1e6:
        return f"¥{x/1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"¥{x/1e3:.1f}K"
    return f"¥{x:.0f}"


# ============ 侧边栏筛选 ============
st.sidebar.markdown("<div class='block-title'>筛选条件</div>", unsafe_allow_html=True)
df = load_data()
REF_DATE = df["Order Date"].max()

regions = sorted(df["Region"].dropna().unique().tolist())
segments = sorted(df["Segment"].dropna().unique().tolist())
years = sorted(df["Order Year"].dropna().unique().tolist())

sel_regions = st.sidebar.multiselect("区域 Region", regions, default=regions)
sel_segments = st.sidebar.multiselect("客群 Segment", segments, default=segments)
sel_years = st.sidebar.multiselect("年份 Year", years, default=years)

if not (sel_regions and sel_segments and sel_years):
    st.warning("请在左侧至少各选一个筛选项。")
    st.stop()

filtered = df[
    df["Region"].isin(sel_regions)
    & df["Segment"].isin(sel_segments)
    & df["Order Year"].isin(sel_years)
].copy()

# ============ 标题栏 ============
st.markdown(
    f"<div class='title-bar'><h1>零售销售数据可视化看板</h1>"
    f"<div class='sub'>数据区间 {filtered['Order Date'].min():%Y-%m-%d} 至 "
    f"{filtered['Order Date'].max():%Y-%m-%d} ｜ 共 {len(filtered):,} 笔订单 ｜ "
    f"当前筛选：{', '.join(sel_regions)} / {', '.join(sel_segments)} / "
    f"{', '.join(map(str, sel_years))}</div></div>",
    unsafe_allow_html=True,
)

# ============ KPI 卡 ============
total_sales = filtered["Sales"].sum()
total_profit = filtered["Profit"].sum()
profit_rate = total_profit / total_sales if total_sales else 0
n_orders = filtered["Order ID"].nunique()
n_customers = filtered["Customer ID"].nunique()
avg_order = total_sales / n_orders if n_orders else 0
kpi = [
    ("总销售额", fmt_money(total_sales), "Sales"),
    ("总利润", fmt_money(total_profit), "Profit"),
    ("利润率", f"{profit_rate:.1%}", "Profit / Sales"),
    ("订单数", f"{n_orders:,}", "Orders"),
    ("客户数", f"{n_customers:,}", "Customers"),
    ("客单价", fmt_money(avg_order), "AOV"),
]
cols = st.columns(6)
for c, (lab, val, sub) in zip(cols, kpi):
    c.markdown(
        f"<div class='kpi'><div class='kpi-v'>{val}</div>"
        f"<div class='kpi-l'>{lab}</div><div class='kpi-s'>{sub}</div></div>",
        unsafe_allow_html=True,
    )

# ============ 六个环形占比（仿看板「占比块」） ============
st.markdown("<div class='block-title'>结构占比</div>", unsafe_allow_html=True)


def doughnut(series, title, top_n=None):
    vc = series.value_counts()
    if top_n:
        if len(vc) > top_n:
            head = vc.head(top_n)
            vc = pd.concat([head, pd.Series({"其他": vc.iloc[top_n:].sum()})])
    fig = px.pie(values=vc.values, names=vc.index, hole=0.58,
                 color_discrete_sequence=DC)
    fig.update_traces(textinfo="percent", textfont=dict(color=LIGHT, family=FONT, size=11),
                      pull=[0.03] * len(vc))
    fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"))
    return dark(fig, h=300, legend=False)


g1, g2, g3 = st.columns(3)
g4, g5, g6 = st.columns(3)
with g1:
    st.plotly_chart(doughnut(filtered["Region"], "区域销售额占比"), width="stretch")
with g2:
    st.plotly_chart(doughnut(filtered["Segment"], "客群销售额占比"), width="stretch")
with g3:
    st.plotly_chart(doughnut(filtered["Category"], "品类销售额占比"), width="stretch")
with g4:
    st.plotly_chart(doughnut(filtered["Ship Mode"], "运输方式占比"), width="stretch")
with g5:
    st.plotly_chart(doughnut(filtered["Sub-Category"], "子类销售额 Top5",
                            top_n=5), width="stretch")
# RFM 八类占比（实时）
cust = (filtered.groupby("Customer ID")
        .agg(R=("Order Date", lambda s: (REF_DATE - s.max()).days),
             F=("Order ID", "nunique"),
             M=("Sales", "sum")).reset_index())
r_med, f_med, m_med = cust["R"].median(), cust["F"].median(), cust["M"].median()
seg = pd.Series(index=cust.index, dtype=object)
seg[(cust.R <= r_med) & (cust.F >= f_med) & (cust.M >= m_med)] = RFM_NAMES[0]
seg[(cust.R > r_med) & (cust.F >= f_med) & (cust.M >= m_med)] = RFM_NAMES[1]
seg[(cust.R <= r_med) & (cust.F < f_med) & (cust.M >= m_med)] = RFM_NAMES[2]
seg[(cust.R > r_med) & (cust.F < f_med) & (cust.M >= m_med)] = RFM_NAMES[3]
seg[(cust.R <= r_med) & (cust.F >= f_med) & (cust.M < m_med)] = RFM_NAMES[4]
seg[(cust.R > r_med) & (cust.F >= f_med) & (cust.M < m_med)] = RFM_NAMES[5]
seg[(cust.R <= r_med) & (cust.F < f_med) & (cust.M < m_med)] = RFM_NAMES[6]
seg[(cust.R > r_med) & (cust.F < f_med) & (cust.M < m_med)] = RFM_NAMES[7]
with g6:
    st.plotly_chart(doughnut(seg, "RFM 客户分层占比"), width="stretch")

# ============ 条形排名 + 面积趋势 ============
st.markdown("<div class='block-title'>排名与趋势</div>", unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)
with b1:
    sub = (filtered.groupby("Sub-Category")["Sales"].sum()
           .sort_values(ascending=True).tail(10))
    fig = go.Figure(go.Bar(x=sub.values, y=sub.index, orientation="h",
                           marker=dict(color=DC[0])))
    fig.update_layout(xaxis=dict(title="销售额"))
    st.plotly_chart(dark(fig, h=320), width="stretch")
with b2:
    reg = filtered.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=reg.index, y=reg.values,
                           marker=dict(color=DC[3])))
    fig.update_layout(yaxis=dict(title="销售额"))
    st.plotly_chart(dark(fig, h=320), width="stretch")
with b3:
    monthly = (filtered.groupby("ym")
               .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
               .reset_index())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["ym"], y=monthly["Sales"].cumsum(),
                            name="累计销售额", fill="tozeroy",
                            line=dict(color=DC[0], width=2.5)))
    fig.update_layout(yaxis=dict(title="累计销售额"))
    st.plotly_chart(dark(fig, h=320), width="stretch")

# ============ 各州地图 + 品类×年份多系列柱 ============
st.markdown("<div class='block-title'>区域分布与品类年度对比</div>",
            unsafe_allow_html=True)
m1, m2 = st.columns([1.3, 1])
state_sales = filtered.groupby("State/Province")["Sales"].sum().reset_index()
state_sales["abbr"] = state_sales["State/Province"].map(STATE_ABBR)
us = state_sales.dropna(subset=["abbr"])
fig_map = px.choropleth(us, locations="abbr", locationmode="USA-states",
                        color="Sales", color_continuous_scale="Blues",
                        scope="usa", hover_name="State/Province")
fig_map.update_layout(coloraxis_colorbar=dict(title="销售额",
                       tickfont=dict(color=LIGHT)))
m1.plotly_chart(dark(fig_map, h=360), width="stretch")
piv = (filtered.pivot_table(index="Category", columns="Order Year",
                            values="Sales", aggfunc="sum").fillna(0))
fig_mb = go.Figure()
for i, y in enumerate(piv.columns):
    fig_mb.add_bar(x=piv.index, y=piv[y], name=str(y),
                   marker=dict(color=DC[i % len(DC)]))
fig_mb.update_layout(barmode="group", yaxis=dict(title="销售额"))
m2.plotly_chart(dark(fig_mb, h=360), width="stretch")

# ============ 订单明细 ============
st.markdown("<div class='block-title'>订单明细（随筛选联动）</div>",
            unsafe_allow_html=True)
detail_cols = ["Order ID", "Order Date", "Region", "Segment", "Ship Mode",
               "Category", "Sub-Category", "Sales", "Profit", "Quantity"]
detail = filtered[detail_cols].sort_values("Order Date", ascending=False)
st.dataframe(detail, width="stretch", height=380)
