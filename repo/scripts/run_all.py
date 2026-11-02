# -*- coding: utf-8 -*-
"""
一键复现流水线：01 取数 -> 02 清洗 -> 03 建模 -> 04 出图
按顺序跑，任一步失败立即停止并报告，不会把半成品继续往下带。

用法（在你本机，仓库根目录）：
    python scripts/run_all.py

说明：
- 05 大屏是 Streamlit 交互页，不在这条批处理链里，单独跑：
    streamlit run scripts/05_dashboard.py
- 所有脚本共用同一套字体/路径兜底，正常在终端或 Jupyter 外运行即可。
- 想看每步打印的关键数字，终端会原样输出（RFM 切分、KMeans 轮廓系数、
  ABC 占比、波士顿象限、同期群留存、SARIMA 预测值等）。
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = ["01_load.py", "02_clean.py", "03_analysis.py", "04_visuals.py"]


def main():
    print(f"仓库根目录: {REPO}\n")
    for name in SCRIPTS:
        script = REPO / "scripts" / name
        if not script.exists():
            print(f"!! 找不到 {name}，请确认 scripts/ 下文件齐全")
            sys.exit(1)
        print("=" * 60)
        print(f">>> 运行 {name}")
        print("=" * 60)
        # 用同一个 python 解释器，cwd 设到仓库根，保证相对路径一致
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(REPO),
        )
        if result.returncode != 0:
            print(f"\n!! {name} 运行失败（返回码 {result.returncode}），已停止。")
            print("   请按上面报错修好后再重跑本脚本。")
            sys.exit(result.returncode)
        print(f"<<< {name} 完成\n")

    print("=" * 60)
    print("全部步骤成功。产物位置：")
    print(f"  清洗数据: {REPO / 'data' / 'clean'}")
    print(f"  模型表格: {REPO / 'outputs' / 'tables'}")
    print(f"  图表    : {REPO / 'outputs' / 'figures'}")
    print("下一步：")
    print("  1) 把 outputs/tables 里的真实数字回填论文【】占位")
    print("  2) 大屏：streamlit run scripts/05_dashboard.py")
    print("  3) 上传 GitHub 并设为 Public")


if __name__ == "__main__":
    main()
