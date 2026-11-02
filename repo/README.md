# 商业数据分析期末大作业

题目：**基于零售交易数据的客户价值与品类结构分析**

数据源：Tableau 官方示例 **Sample - Superstore**（Orders / People / Returns 三张表）

## 目录结构

```
repo/
├── data/
│   ├── raw/            # 原始文件：Sample - Superstore.xls / .xlsx
│   └── clean/          # 清洗后 CSV + SQLite
├── scripts/
│   ├── 01_load.py      # 取数：读三张表，导出 CSV / SQLite
│   ├── 02_clean.py     # 清洗：类型/缺失/异常/重复/派生/多表对齐
│   ├── 03_analysis.py  # 六模型：RFM / K-Means / ABC / 波士顿 / 同期群 / SARIMA
│   ├── 04_visuals.py   # 出图：12 张模板复现 + 6 张业务图
│   └── 05_dashboard.py # 大屏：Streamlit 6 组件联动
├── outputs/
│   ├── figures/        # 所有 PNG 图（fig01~fig18）
│   └── tables/         # 模型结果表
├── docs/               # 论文 + 数据来源合规声明
├── requirements.txt
├── run_all.py          # 一键复现（顺序跑 01~05）
└── README.md
```

## 复现步骤

```bash
pip install -r requirements.txt
python run_all.py
# 或分步：
python scripts/01_load.py
python scripts/02_clean.py
python scripts/03_analysis.py
python scripts/04_visuals.py
streamlit run scripts/05_dashboard.py
```

## 图表索引

- fig01~fig12：模板复现（对应教材《Excel 数据可视化》第二章）
- fig13~fig18：业务模型图（RFM / K-Means / ABC / 波士顿 / 同期群 / SARIMA）

## 合规说明

- 数据来源：Tableau Public 官方示例 “Sample - Superstore”
- 仅使用数据，未使用 Tableau 软件做图
- 全部图表由 Python 生成，无爬虫、无 BI 软件、无 Excel 截图
