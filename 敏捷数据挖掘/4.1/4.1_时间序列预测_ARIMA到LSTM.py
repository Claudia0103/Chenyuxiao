# -*- coding: utf-8 -*-
"""
4.1 时间序列预测 —— 合并教材 4 个 notebook（ARIMA / SARIMA / 趋势分解 / LSTM）
==========================================================================
  1) ts_simu200.csv：平稳序列 AR(1) / MA(1) / ARMA(1,1)、非平稳 ARIMA(1,1,0)；
     ACF/PACF 定阶、ADF 平稳性检验、AIC 网格、残差 DW 与正态性检验、预测；
  2) tractor_sales.csv：月度销量 → 对数变换 → 季节性分解 → SARIMA 网格定阶
     → 残差诊断 → 36 期预测与 95% 置信区间；
  3) air.csv：航空旅客数 → 加法/乘法分解 → Holt-Winters 指数平滑预测；
  4) google_stock_prics.csv：滑动窗口构造样本 → 神经网络回归预测股价，
     与"用昨天价格当预测"的基线比较 RMSE / MAE。

两个替代（环境原因，写清楚）：
  - fbprophet 是已停止维护的老包、需要 C++ 编译 → 用 statsmodels 的
    seasonal_decompose + ExponentialSmoothing(Holt-Winters) 复现"加法/乘法季节模型 +
    区间预测"这一内涵；
  - tensorflow / keras 体积大且本环境未装 → 用 sklearn MLPRegressor 在同样的
    滑动窗口样本上做回归，演示"神经网络做多步特征 → 一步预测"的流程。

数据：data/ts_simu200.csv、tractor_sales.csv、air.csv、google_stock_prics.csv。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('air.csv', 'google_stock_prics.csv', 'tractor_sales.csv', 'ts_simu200.csv')   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\敏捷数据挖掘复现\4.1时间序列预测"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '4.1时间序列预测'),
              os.path.join(os.getcwd(), '敏捷数据挖掘复现', '4.1时间序列预测')]
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
    p = os.path.join(CHARTS, f"4.1_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


# ---------------------------------------------------------------- 1 ARIMA 家族
def arma_family():
    d = pd.read_csv(os.path.join(DATA, "ts_simu200.csv"), index_col="t")
    dates = pd.date_range("2017-01-01", periods=len(d), freq="D")
    d.index = dates
    show("模拟序列字段", list(d.columns))
    series = {"AR(1)": ("AR1_a", (1, 0, 0)),
              "MA(1)": ("MA1_a", (0, 0, 1)),
              "ARMA(1,1)": ("ARMA_11_b", (1, 0, 1)),
              "ARIMA(1,1,0)": ("ARIMA_110", (1, 1, 0))}
    out = []
    for label, (col, order) in series.items():
        y = d[col]
        adf_p = sm.tsa.stattools.adfuller(y)[1]
        m = sm.tsa.ARIMA(y, order=order).fit()
        resid = m.resid
        out.append({"模型": label, "ADF p值": adf_p, "AIC": m.aic,
                    "DW": sm.stats.durbin_watson(resid.values),
                    "残差正态 p": stats.normaltest(resid).pvalue,
                    "5步预测均值": round(float(m.forecast(5).mean()), 4)})
        print(f"  {label} 拟合完成")
    show("ARIMA 家族拟合与残差诊断", pd.DataFrame(out).round(4).to_string(index=False))

    y = d["ARMA_11_b"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    sm.graphics.tsa.plot_acf(y, lags=20, ax=ax[0])
    sm.graphics.tsa.plot_pacf(y, lags=20, ax=ax[1])
    ax[0].set_title("ACF"); ax[1].set_title("PACF")
    fig.tight_layout()
    save(fig, "ARMA_ACF_PACF")

    # AIC 网格定阶
    best = (np.inf, None)
    for p, q in itertools.product(range(3), range(3)):
        try:
            aic = sm.tsa.ARIMA(y, order=(p, 0, q)).fit().aic
            if aic < best[0]:
                best = (aic, (p, 0, q))
        except Exception:
            continue
    show("ARMA 网格定阶（最小 AIC）", f"order={best[1]}, AIC={best[0]:.2f}")


# ---------------------------------------------------------------- 2 SARIMA
def sarima_sales():
    d = pd.read_csv(os.path.join(DATA, "tractor_sales.csv"))
    idx = pd.to_datetime(d["Year_Month"], format="%Y-%b")
    ts = pd.Series(d["Number_of_Sold"].values, index=idx, name="sales").astype(float)
    show("拖拉机销量序列", f"{len(ts)} 期，{ts.index[0].date()} ~ {ts.index[-1].date()}")
    ts = ts.asfreq("MS").interpolate()

    dec = sm.tsa.seasonal_decompose(ts, model="multiplicative")
    fig = dec.plot()
    fig.set_size_inches(9, 6)
    save(fig, "销量季节性分解")

    log_ts = np.log(ts)
    best = (np.inf, None, None)
    for order in itertools.product(range(2), [1], range(2)):
        for sorder in itertools.product(range(2), [1], range(2)):
            so = tuple(sorder) + (12,)
            try:
                m = sm.tsa.statespace.SARIMAX(log_ts, order=order,
                                             seasonal_order=so).fit(disp=False)
                if m.aic < best[0]:
                    best = (m.aic, order, so)
            except Exception:
                continue
    show("SARIMA 网格定阶", f"order={best[1]}, seasonal={best[2]}, AIC={best[0]:.2f}")

    model = sm.tsa.statespace.SARIMAX(log_ts, order=best[1],
                                     seasonal_order=best[2]).fit(disp=False)
    show("模型摘要（前 8 行）", str(model.summary().tables[1])[:600])
    fc = model.get_forecast(steps=36)
    mean = np.exp(fc.predicted_mean)
    ci = np.exp(fc.conf_int())
    fidx = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1), periods=36, freq="MS")
    mean.index, ci.index = fidx, fidx

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ts.index, ts.values, label="观测值", color="#5aa9ff")
    ax.plot(mean.index, mean.values, "--", label="36 期预测", color="#ff7b7b")
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1],
                    color="#ff7b7b", alpha=.15, label="95% 置信区间")
    ax.set(title="SARIMA 拖拉机销量预测", xlabel="月份", ylabel="销量")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    save(fig, "SARIMA销量预测")
    show("预测前 5 期", mean.head().round(1))


# ---------------------------------------------------------------- 3 趋势分解 / Holt-Winters
def trend_air():
    d = pd.read_csv(os.path.join(DATA, "air.csv"))
    d["DATE"] = pd.to_datetime(d["DATE"])
    ts = d.set_index("DATE")["AIR"].astype(float).asfreq("MS")
    show("航空旅客序列", f"{len(ts)} 期，{ts.index[0].date()} ~ {ts.index[-1].date()}")

    results = {}
    for mode in ("additive", "multiplicative"):
        dec = sm.tsa.seasonal_decompose(ts, model=mode, period=12)
        resid_std = float(np.nanstd(dec.resid))
        hw = ExponentialSmoothing(ts, trend="add", seasonal=mode,
                                  seasonal_periods=12).fit()
        pred = hw.forecast(36)
        mse = float(np.mean((hw.fittedvalues - ts) ** 2))
        results[mode] = {"残差标准差": resid_std, "拟合MSE": mse,
                         "未来12期均值": round(float(pred[:12].mean()), 1)}
        if mode == "multiplicative":
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(ts.index, ts.values, label="观测值", color="#5aa9ff")
            ax.plot(pred.index, pred.values, "--", label="Holt-Winters 预测",
                    color="#3ddc97")
            ax.set(title="航空旅客：乘法季节模型预测（Prophet 等价替代）",
                   xlabel="日期", ylabel="旅客数")
            ax.legend(); ax.grid(alpha=.3)
            fig.tight_layout()
            save(fig, "航空旅客预测")
            fig2 = dec.plot()
            fig2.set_size_inches(9, 6)
            save(fig2, "航空旅客乘法分解")
    show("加法 vs 乘法季节模型", pd.DataFrame(results).T.round(2))


# ---------------------------------------------------------------- 4 神经网络预测股价
def sliding_windows(arr, timesteps=20):
    X = np.array([arr[i:i + timesteps].ravel() for i in range(len(arr) - timesteps)])
    y = np.array([arr[i + timesteps, 1] for i in range(len(arr) - timesteps)])
    return X, y


def nn_stock():
    d = pd.read_csv(os.path.join(DATA, "google_stock_prics.csv"))
    price = d[["Open", "Close"]].astype(float)
    cut = int(len(price) * .8)
    train, test = price.iloc[:cut].copy(), price.iloc[cut:].copy()
    scaler = MinMaxScaler().fit(train)
    tr, te = scaler.transform(train), scaler.transform(test)
    show("股价数据", f"训练 {len(train)} 期 / 测试 {len(test)} 期")

    Xtr, ytr = sliding_windows(tr)
    Xte, yte = sliding_windows(te)
    # 基线：直接用上一期的 Close 当预测
    baseline = np.array([te[i + 20 - 1, 1] for i in range(len(yte))])
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu",
                       max_iter=400, random_state=0)
    mlp.fit(Xtr, ytr)
    pred = mlp.predict(Xte)
    show("测试集 RMSE / MAE（标准化尺度）", pd.DataFrame({
        "基线(上一日收盘)": [np.sqrt(mean_squared_error(yte, baseline)),
                          mean_absolute_error(yte, baseline)],
        "MLP 滑动窗口": [np.sqrt(mean_squared_error(yte, pred)),
                      mean_absolute_error(yte, pred)]},
        index=["RMSE", "MAE"]).round(4))

    close_scaled = te[20:, 1]
    pred_price = scaler.inverse_transform(
        np.c_[np.zeros(len(pred)), pred])[:, 1]
    base_price = scaler.inverse_transform(
        np.c_[np.zeros(len(baseline)), baseline])[:, 1]
    real_price = test["Close"].values[20:]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(real_price, label="真实收盘价", color="#5aa9ff")
    ax.plot(pred_price, "--", label="MLP 预测", color="#ff7b7b", lw=1.2)
    ax.plot(base_price, ":", label="基线", color="gray", lw=1)
    ax.set(title="Google 股价预测（LSTM 的轻量等价复现）",
           xlabel="测试期序号", ylabel="价格")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    save(fig, "股价预测")
    show("原始尺度 RMSE", pd.Series({
        "基线": np.sqrt(mean_squared_error(real_price, base_price)),
        "MLP": np.sqrt(mean_squared_error(real_price, pred_price))}).round(2))


def main():
    print("4.1 时间序列预测：ARIMA → SARIMA → 趋势分解 → 神经网络")
    arma_family()
    sarima_sales()
    trend_air()
    nn_stock()


if __name__ == "__main__" or True:
    main()
