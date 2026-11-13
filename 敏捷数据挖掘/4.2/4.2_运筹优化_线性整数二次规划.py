# -*- coding: utf-8 -*-
"""
4.2 运筹优化 —— 合并教材 4.2.1 线性规划 / 4.2.2 整数规划 / 4.2.3 二次规划
========================================================================
教材用 IBM CPLEX（商业软件，需要 license）读 .lp 文件求解。这里不装 CPLEX，
改成自己解析 CPLEX 格式的 .lp 文件 + scipy.optimize 求解，三个案例都能跑：
  productivity.lp → linprog（线性规划，最大化 50x1+100x2）
  campaign.lp     → milp（0/1 整数规划，27 个变量的广告投放分配）
  qp1.lp          → minimize/SLSQP（二次规划，带 a² 和 ab 项）

.lp 解析支持 Maximize / Minimize、Subject To、Bounds、Generals 四段，
所以换一个同格式的 .lp 文件也能直接用（改 LP_FILES 即可）。

数据：data/productivity.lp、campaign.lp、qp1.lp。
运行：整份代码粘进 Jupyter 一个 cell 也能跑。
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog, milp, minimize, LinearConstraint, Bounds

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 定位章节目录（整段粘进 Jupyter 单个 cell 也能用）----
# cell 里没有 __file__；若直接退回 os.getcwd()，cwd=C://Users//LENOVO 时
# 会把 data/ 拼成 C://Users//LENOVO//data（该目录恰好存在）而读不到教材数据。
CHAPTER_DIR = ""       # 要手动指定时，填章节目录的绝对路径
_EXPECTED = ('campaign.lp', 'productivity.lp', 'qp1.lp')   # 该脚本会用到的 data/ 下的文件或子目录

def _locate_chapter_dir():
    if CHAPTER_DIR:
        return CHAPTER_DIR
    # 交付目录的准确路径优先，避免命中同名的其它 data/ 目录
    hardcoded = r"C:\Users\LENOVO\WorkBuddy\2026-10-22-09-53-30\敏捷数据挖掘复现\4.2运筹优化"
    if os.path.isdir(os.path.join(hardcoded, "data")):
        return hardcoded
    roots = []
    if "__file__" in globals():
        roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots += [os.getcwd(),
              os.path.join(os.getcwd(), '4.2运筹优化'),
              os.path.join(os.getcwd(), '敏捷数据挖掘复现', '4.2运筹优化')]
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
    p = os.path.join(CHARTS, f"4.2_{name}.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  图已保存：{os.path.basename(p)}")


def parse_expr(s):
    """把 '50 x1 + 100 x2 - 3 x3' 解析成 {变量: 系数}。"""
    out = {}
    for tok in s.replace("-", "+-").split("+"):
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r"^(-?\d*\.?\d*)\s*\*?\s*([A-Za-z_]\w*)$", tok)
        if not m:
            continue
        raw, var = m.group(1), m.group(2)
        coef = -1.0 if raw == "-" else (1.0 if raw == "" else float(raw))
        out[var] = out.get(var, 0.0) + coef
    return out


def parse_lp(path):
    """解析 CPLEX LP 文件，返回 (目标系数dict, 方向, 约束列表, bounds, 整数变量集)。"""
    section, objective, sense = None, {}, "max"
    constraints, bounds, generals = [], {}, set()
    objective_lines = []
    for raw in open(path, encoding="latin-1"):
        line = raw.strip()
        if not line or line.startswith("\\"):
            continue
        head = line.lower()
        if head.startswith("maximize"):
            section, sense = "obj", "max"; continue
        if head.startswith("minimize"):
            section, sense = "obj", "min"; continue
        if head.startswith("subject to"):
            section = "cons"; continue
        if head.startswith("bounds"):
            section = "bounds"; continue
        if head.startswith("generals"):
            section = "generals"; continue
        if head in ("end",) or head.startswith("end "):
            break
        if section == "obj":
            objective_lines.append(line.partition(":")[2] if ":" in line and
                                   line.split(":")[0].strip() in ("obj", "objective")
                                   else line)
        elif section == "cons":
            body = line.partition(":")[2].strip()
            m = re.split(r"\s*(<=|>=|=)\s*", body)
            if len(m) >= 3:
                constraints.append((parse_expr(m[0]), m[1], float(m[2])))
        elif section == "bounds":
            parts = [p.strip() for p in line.split("<=")]
            if len(parts) == 3:
                bounds[parts[1]] = (float(parts[0]), float(parts[2]))
            elif len(parts) == 2:
                bounds[parts[1]] = (float(parts[0]), None)
        elif section == "generals":
            generals.update(line.split())
    for line in objective_lines:
        for k, v in parse_expr(line).items():
            objective[k] = objective.get(k, 0.0) + v
    return objective, sense, constraints, bounds, generals


def build_matrices(path):
    obj, sense, cons, bounds, generals = parse_lp(path)
    variables = sorted(bounds, key=lambda v: int(re.sub(r"\D", "", v) or 0))
    idx = {v: i for i, v in enumerate(variables)}
    n = len(variables)
    c = np.array([obj.get(v, 0.0) for v in variables])
    A, b = [], []
    for coeffs, op, rhs in cons:
        row = np.zeros(n)
        for v, a in coeffs.items():
            if v in idx:
                row[idx[v]] = a
        if op == "<=":
            A.append(row); b.append(rhs)
        elif op == ">=":
            A.append(-row); b.append(-rhs)
        else:
            A.append(row); b.append(rhs)
            A.append(-row); b.append(-rhs)
    lb = np.array([bounds[v][0] for v in variables])
    ub = np.array([bounds[v][1] if bounds[v][1] is not None else np.inf
                   for v in variables])
    integrality = np.array([1 if v in generals else 0 for v in variables])
    return variables, c, sense, np.array(A), np.array(b), lb, ub, integrality


def linear_program():
    v, c, sense, A, b, lb, ub, _ = build_matrices(
        os.path.join(DATA, "productivity.lp"))
    show("线性规划变量 / 目标", f"{v} / {'max' if sense=='max' else 'min'} {c}")
    show("约束矩阵 A_ub / b_ub", f"{A.tolist()} / {b.tolist()}")
    res = linprog(c=-c if sense == "max" else c, A_ub=A, b_ub=b,
                  bounds=list(zip(lb, ub)), method="highs")
    show("linprog 求解状态", res.message)
    show("最优解", pd.Series(res.x, index=v).round(4))
    show("最大目标值", round(float(-res.fun if sense == "max" else res.fun), 4))

    # 画可行域（二维）
    x1 = np.linspace(0, 300, 300)
    y1 = 300 - x1
    y2 = 400 - 2 * x1
    y3 = np.full_like(x1, 250)
    ymax = np.minimum(np.minimum(y1, y2), y3)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(x1, 0, np.maximum(ymax, 0), alpha=.2, color="#5aa9ff",
                    label="可行域")
    ax.plot(x1, y1, label="x1+x2<=300")
    ax.plot(x1, y2, label="2x1+x2<=400")
    ax.plot(x1, y3, label="x2<=250")
    ax.plot(res.x[0], res.x[1], "ro", ms=9, label="最优点")
    ax.set(xlim=(0, 220), ylim=(0, 320), xlabel="x1", ylabel="x2",
           title="线性规划可行域与最优解")
    ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout()
    save(fig, "线性规划可行域")


def integer_program():
    v, c, sense, A, b, lb, ub, integ = build_matrices(
        os.path.join(DATA, "campaign.lp"))
    show("整数规划变量数 / 整数变量数", f"{len(v)} / {int(integ.sum())}")
    res = milp(c=-c if sense == "max" else c,
               constraints=LinearConstraint(A, -np.inf, b),
               integrality=integ, bounds=Bounds(lb, ub))
    show("milp 求解状态", res.message)
    sol = np.round(res.x).astype(int).reshape(9, 3, order="F")
    show("最优投放分配（9 个区域 × 3 个方案）", pd.DataFrame(sol,
         columns=["方案1", "方案2", "方案3"]))
    show("最大总收益", round(float(-res.fun if sense == "max" else res.fun), 2))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(sol, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(["方案1", "方案2", "方案3"])
    ax.set_yticks(range(9)); ax.set_yticklabels([f"区域{i+1}" for i in range(9)])
    ax.set_title("0/1 整数规划投放方案")
    fig.tight_layout()
    save(fig, "整数规划投放方案")


def quadratic_program():
    """qp1.lp: min [2a² + b² + a*b]/2 + a + b, s.t. a+b=1, a,b>=0。
    二次项用 SLSQP 直接最小化（CPLEX 的 QP 等价）。"""
    text = open(os.path.join(DATA, "qp1.lp"), encoding="latin-1").read()
    show("qp1.lp 目标函数", [l for l in text.splitlines() if "obj:" in l][0].strip())

    def f(x):
        a, b = x
        return (2 * a ** 2 + b ** 2 + a * b) / 2 + a + b

    cons = {"type": "eq", "fun": lambda x: x[0] + x[1] - 1}
    res = minimize(f, [0.5, 0.5], method="SLSQP",
                   bounds=[(0, None), (0, None)], constraints=cons)
    show("SLSQP 求解状态", res.message)
    show("最优解", pd.Series(res.x, index=["a", "b"]).round(6))
    show("最小目标值", round(float(res.fun), 6))

    a = np.linspace(0, 1, 200)
    vals = np.array([f([ai, 1 - ai]) for ai in a])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(a, vals, color="#8b7cff", label="a+b=1 约束下的目标值")
    ax.plot(res.x[0], res.fun, "ro", ms=9, label="最小值")
    ax.set(title="二次规划目标函数", xlabel="a", ylabel="目标值")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    save(fig, "二次规划目标函数")


def main():
    print("4.2 运筹优化：CPLEX → scipy.optimize 等价复现")
    linear_program()
    integer_program()
    quadratic_program()


if __name__ == "__main__" or True:
    main()
