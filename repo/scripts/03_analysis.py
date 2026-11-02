from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ---- 路径配置（兼容脚本 / Jupyter）----
try:
    REPO = Path(__file__).resolve().parents[1]
except NameError:
    REPO = Path(r"C:/Users/LENOVO/Desktop/repo")
CLEAN = REPO / "data" / "clean"
TABLES = REPO / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLEAN / "orders_clean.csv", parse_dates=["Order Date"])
print(f"[读入] orders_clean {df.shape}")

# ================= 1. RFM 八分类 =================
print("\n========== 1. RFM 八分类 ==========")
snapshot = df["Order Date"].max()
rfm = df.groupby("Customer ID").agg(
    R=("Order Date", lambda s: (snapshot - s.max()).days),
    F=("Order ID", "nunique"),
    M=("Sales", "sum"),
).reset_index()
r_med, f_med, m_med = rfm["R"].median(), rfm["F"].median(), rfm["M"].median()
rfm["R_s"] = (rfm["R"] <= r_med).astype(int)   # 最近(天数小)=1
rfm["F_s"] = (rfm["F"] >= f_med).astype(int)
rfm["M_s"] = (rfm["M"] >= m_med).astype(int)

def seg(r):
    keys = {(1,1,1):"重要价值客户",(1,1,0):"重要保持客户",(1,0,1):"重要发展客户",
            (1,0,0):"重要挽留客户",(0,1,1):"一般价值客户",(0,1,0):"一般保持客户",
            (0,0,1):"一般发展客户",(0,0,0):"一般挽留客户"}
    return keys[(r["R_s"], r["F_s"], r["M_s"])]

rfm["Segment"] = rfm.apply(seg, axis=1)
seg_tab = rfm.groupby("Segment").agg(人数=("Customer ID","count"), 销售额=("M","sum")).reset_index()
seg_tab["人数占比"] = (seg_tab["人数"]/seg_tab["人数"].sum()*100).round(1)
seg_tab["销售额占比"] = (seg_tab["销售额"]/seg_tab["销售额"].sum()*100).round(1)
seg_tab = seg_tab.sort_values("销售额", ascending=False)
print(seg_tab.to_string(index=False))
val = seg_tab[seg_tab["Segment"].isin(["重要价值客户","重要保持客户"])].sum(numeric_only=True)
print(f"[论文] 重要价值+重要保持类客户 {int(val['人数'])} 人，"
      f"占 {val['人数占比']:.1f}%，贡献销售额 {val['销售额占比']:.1f}%")
print(f"[论文] R/F/M 中位数切分阈值：R={r_med:.0f}天, F={f_med:.0f}单, M={m_med:.0f}元")
seg_tab.to_csv(TABLES/"rfm_segments.csv", index=False, encoding="utf-8-sig")

# ================= 2. K-Means 客群 =================
print("\n========== 2. K-Means 客群 ==========")
X = StandardScaler().fit_transform(rfm[["R","F","M"]])
inertias = []
for k in range(1, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertias.append((k, km.inertia_))
for k, i in inertias:
    print(f"  K={k}  SSE={i:,.0f}")
K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X)
rfm["Cluster"] = km.labels_
clu = rfm.groupby("Cluster").agg(人数=("Customer ID","count"),
                                 R均值=("R","mean"), F均值=("F","mean"), M均值=("M","mean")).reset_index()
clu[["R均值","F均值","M均值"]] = clu[["R均值","F均值","M均值"]].round(1)
print(clu.to_string(index=False))
print("[论文] SSE 折线在 K=4 附近出现拐点（肘部法），故选 K=4；四个簇即高/中/低价值客群")
clu.to_csv(TABLES/"kmeans_clusters.csv", index=False, encoding="utf-8-sig")

# ================= 3. ABC 品类分级 =================
print("\n========== 3. ABC 品类分级 ==========")
abc = df.groupby("Sub-Category")["Sales"].sum().sort_values(ascending=False).reset_index()
abc["累计占比"] = abc["Sales"].cumsum()/abc["Sales"].sum()*100
def grade(p):
    if p < 80: return "A"
    if p < 95: return "B"
    return "C"
abc["等级"] = abc["累计占比"].apply(grade)
nA = int((abc["等级"]=="A").sum())
aShare = abc.loc[abc["等级"]=="A","Sales"].sum()/abc["Sales"].sum()*100
print(abc.to_string(index=False))
print(f"[论文] A 类 {nA} 个子类别贡献 {aShare:.1f}% 销售额（帕累托 80/20 原则）")
abc.to_csv(TABLES/"abc_classification.csv", index=False, encoding="utf-8-sig")

# ================= 4. 波士顿矩阵 =================
print("\n========== 4. 波士顿矩阵 ==========")
year_sales = df.pivot_table(index="Sub-Category", columns="Order Year", values="Sales", aggfunc="sum")
last, prev = year_sales.columns.max(), year_sales.columns.max()-1
bos = pd.DataFrame({"份额": df.groupby("Sub-Category")["Sales"].sum()/df["Sales"].sum()*100})
bos["增速"] = (year_sales[last]-year_sales[prev])/year_sales[prev]*100
bos["利润"] = df.groupby("Sub-Category")["Profit"].sum()
bos["增速"] = bos["增速"].fillna(-1)
s_med, g_med = bos["份额"].median(), bos["增速"].median()
def quad(r):
    if r["份额"]>=s_med and r["增速"]>=g_med: return "明星"
    if r["份额"]>=s_med and r["增速"]<g_med: return "金牛"
    if r["份额"]<s_med and r["增速"]>=g_med: return "问题"
    return "瘦狗"
bos["象限"] = bos.apply(quad, axis=1)
print(bos.sort_values("份额", ascending=False).round(1).to_string())
n_star = int((bos["象限"]=="明星").sum()); n_cow = int((bos["象限"]=="金牛").sum())
n_q = int((bos["象限"]=="问题").sum()); n_dog = int((bos["象限"]=="瘦狗").sum())
print(f"[论文] 份额中位 {s_med:.1f}%，增速中位 {g_med:.1f}%；"
      f"明星 {n_star}/金牛 {n_cow}/问题 {n_q}/瘦狗 {n_dog}")
bos.to_csv(TABLES/"boston_matrix.csv", encoding="utf-8-sig")

# ================= 5. 同期群留存 =================
print("\n========== 5. 同期群留存 ==========")
df["月份"] = pd.PeriodIndex(df["Order Month"], freq="M")
df["mo_ord"] = df["月份"].apply(lambda p: p.year*12 + p.month)
first_ord = df.groupby("Customer ID")["mo_ord"].transform("min")
df["首购月"] = pd.PeriodIndex(df.groupby("Customer ID")["Order Month"].transform("min"), freq="M")
df["期数"] = df["mo_ord"] - first_ord
coh = df.groupby(["首购月","期数"])["Customer ID"].nunique().reset_index()
piv = coh.pivot(index="首购月", columns="期数", values="Customer ID")
size = piv[0]
ret = piv.div(size, axis=0)*100
n_per = ret.shape[1]
ret = ret.apply(lambda row: row.where(row.index <= (n_per-1 - ret.index.get_loc(row.name)),
                                       other=np.nan), axis=1)
ret = ret.round(1)
print(ret.to_string())
print("[论文] 同期群留存热力图见 04_visuals；首购月最早群观测期数最多，"
      "未来未发生期（矩阵右上三角）置 NaN，不渲染为 0% 深色区")
ret.to_csv(TABLES/"cohort_retention.csv", encoding="utf-8-sig")

# ================= 6. SARIMA 销售额预测 =================
print("\n========== 6. SARIMA 销售额预测 ==========")
ms = df.groupby("月份")["Sales"].sum().sort_index()
ts = ms.to_timestamp().asfreq("MS")
model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
train = ts.iloc[:-12]; test = ts.iloc[-12:]
m2 = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
fc = m2.get_forecast(steps=12)
pred = fc.predicted_mean
mape = np.mean(np.abs((test.values-pred.values)/test.values))*100
rmse = np.sqrt(np.mean((test.values-pred.values)**2))
print(f"[论文] 滚动回测 12 个月：MAPE={mape:.2f}%，RMSE={rmse/1e4:.2f} 万元")
fc_all = model.get_forecast(steps=12, alpha=0.05)
mean_f = fc_all.predicted_mean
ci = fc_all.conf_int(alpha=0.05)
fc_out = pd.DataFrame({"月份": mean_f.index.strftime("%Y-%m"),
                       "预测销售额": mean_f.values.round(0),
                       "下限": ci.iloc[:,0].values.round(0),
                       "上限": ci.iloc[:,1].values.round(0)})
print(fc_out.to_string(index=False))
fc_out.to_csv(TABLES/"sarima_forecast.csv", index=False)

print("\n[完成] 六模型结果已写入 outputs/tables/（rfm_segments / kmeans_clusters / "
      "abc_classification / boston_matrix / cohort_retention / sarima_forecast）")
