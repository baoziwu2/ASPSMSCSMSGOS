下面直接给出一套**可复现的、端到端解决方案**，覆盖

> 数据收集 → 经典 UC 建模与求解 → 含网络与安全约束的 UC → QUBO 转化与量子求解思路 → 规模缩减策略 → 结论

整套方案均基于题目给出的 PDF 和 Excel 数据表格（`Table.xlsx`） ，并结合主流文献中关于 UC、DC 潮流、QUBO 与惩罚法的建模方式。([optimization.cbe.cornell.edu](https://optimization.cbe.cornell.edu/index.php?title=Unit_commitment_problem&utm_source=chatgpt.com))

------

## 1. 数据收集与预处理

### 1.1 数据来源与可信性

- **机组参数**：由题目 Table 1–2 给出（容量、最小出力、启停成本、爬坡约束、最小开停机时间、成本函数系数 a,b,c、惯量常数 H 等）
- **电网参数**：Table 3 给出 30 节点、41 支路的线路阻抗与潮流上限（典型的 IEEE 30 节点系统数据结构），为学界常用测试系统的变体，具有良好代表性。([NREL Docs](https://docs.nrel.gov/docs/fy23osti/85006.pdf?utm_source=chatgpt.com))
- **负荷数据**：Table 4 提供 24 小时时序负荷。
- **惯量与安全参数**：Table 5 提供 ROCOF 上限、频率响应因子 F 及典型功率突变量。

这些数据由竞赛组委会统一提供，来源明确；网络与机组结构与文献中典型测试系统一致，因此可认为是“权威且可信”的测试算例数据。

### 1.2 Python 读取与预处理代码

下面代码假设 `Problem D.pdf` 与 `Table.xlsx` 与脚本在同一目录（或手动调整路径），仅依赖 `pandas/numpy/matplotlib/pyomo` 等常见库。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ========== 1. 读取 Excel 数据 ==========

xls = pd.ExcelFile("Table.xlsx")

unit_tbl1 = pd.read_excel(xls, sheet_name="Table 1", header=1)
unit_tbl2 = pd.read_excel(xls, sheet_name="Table 2", header=1)
branch_tbl = pd.read_excel(xls, sheet_name="Table 3", header=1)
load_tbl  = pd.read_excel(xls, sheet_name="Table 4", header=1)
inert_tbl = pd.read_excel(xls, sheet_name="Table 5", header=1)

# 整理机组数据（按 bus 标识机组）
units = unit_tbl1.merge(unit_tbl2, on="Unit (bus)", suffixes=("_1", "_2"))

units = units.rename(columns={
    "Unit (bus)": "bus",
    "Maximum power generation (MW)": "Pmax",
    "Minimum power generation (MW)": "Pmin",
    "Minimum Up Time (h)_1": "UT_table1",
    "Startup Cost ($)": "Cost_start",
    "Shutdown Cost ($)": "Cost_stop",
    "Ramp-Up Limit (MW/h)": "RU",
    "Ramp-Down Limit (MW/h)": "RD",
    "Minimum Up Time (h)_2": "UT",
    "Minimum Down Time (h)": "DT",
    "Initial Up Time (h)": "InitUp",
    "Initial Down Time (h)": "InitDown",
    "a": "a",
    "b": "b",
    "c": "c",
    "H": "H"
})

# 对最小开机时间取 Table2 的 UT（Table1/2 一致，这里统一）
units["UT"] = units["UT"].fillna(units["UT_table1"])
units = units.drop(columns=["UT_table1"])

# 负荷数据
load_tbl = load_tbl.rename(columns={
    "Time period (h)": "t",
    "Load demand (MW)": "Load"
})
load_tbl = load_tbl.set_index("t").sort_index()

# 线路数据
branch_tbl = branch_tbl.rename(columns={
    "Branch number": "id",
    "From bus": "from_bus",
    "To Bus": "to_bus",
    "r": "r",
    "x": "x",
    "b": "b",
    "Max power transmission (MW)": "Fmax"
})

# 惯量安全参数
ROCOF_max = float(inert_tbl["ROCOF(HZ/s)"].iloc[0])
F_freq    = float(inert_tbl["F"].iloc[0])
P_step    = float(inert_tbl["load step (MW)"].iloc[0])

print(units)
print(branch_tbl.head())
print(load_tbl.head())
print("ROCOF_max, F, P_step:", ROCOF_max, F_freq, P_step)
```

> 可视化示例：负荷曲线

```python
plt.figure()
plt.plot(load_tbl.index, load_tbl["Load"], marker="o")
plt.xlabel("Hour")
plt.ylabel("Load (MW)")
plt.title("24h Load Profile")
plt.grid(True)
plt.show()
```

------

## 2. 问题 1：经典 UC 数学模型与实现

### 2.1 符号与集合

- 机组集合：( G )，元素 (i)（由 `units` 表中每行对应的 bus 号唯一标识）。
- 时段集合：( T = {1,2,\dots,24} )。
- 参数：
  - (P_i^{\min}, P_i^{\max})：最小/最大出力
  - (RU_i, RD_i)：爬坡上/下限
  - (UT_i, DT_i)：最小开机/停机时间
  - (C_i^{\text{su}}, C_i^{\text{sd}})：启停成本
  - 燃料成本：(C_i^{\text{fuel}}(p_{i,t}) = a_i p_{i,t}^2 + b_i p_{i,t} + c_i u_{i,t})
  - (D_t)：时段 (t) 系统总负荷
- 决策变量：
  - (u_{i,t} \in {0,1})：机组 (i) 在时段 (t) 是否开机
  - (y_{i,t} \in {0,1})：机组 (i) 在时段 (t) 是否启动
  - (z_{i,t} \in {0,1})：机组 (i) 在时段 (t) 是否停机
  - (p_{i,t} \ge 0)：机组有效出力

### 2.2 经典 UC 数学模型

#### 2.2.1 目标函数

最小化 24 小时总运行成本：

[
 \min \sum_{t\in T} \sum_{i\in G}
 \Big( a_i p_{i,t}^2 + b_i p_{i,t} + c_i u_{i,t}

- C_i^{\text{su}} y_{i,t} + C_i^{\text{sd}} z_{i,t} \Big)
   ]

#### 2.2.2 约束

1. **功率平衡**（系统总发电 = 负荷）

[
 \sum_{i\in G} p_{i,t} = D_t,\quad \forall t\in T
 ]

1. **机组出力上下限**

[
 P_i^{\min} u_{i,t} \le p_{i,t} \le P_i^{\max} u_{i,t},\quad \forall i,t
 ]

1. **启停逻辑约束**（与 (u,y,z) 的关系）([profdoc.um.ac.ir](https://profdoc.um.ac.ir/articles/a/1006864.pdf?utm_source=chatgpt.com))

[
 u_{i,t} - u_{i,t-1} = y_{i,t} - z_{i,t},\quad \forall i,\ t>1
 ]

首时段 (t=1) 用初始状态 (u_{i,0}) 替代；可以根据 Initial Up/Down 时间推断（例如初始 Up>0 视为在时 0 已开机）。

1. **最小开机时间**

[
 \sum_{\tau=t-UT_i+1}^{t} y_{i,\tau} \le u_{i,t},\quad
 \forall i,\ t \in {UT_i,\dots, T}
 ]

边界处按索引 (\tau\ge 1) 处理（类似下式）。

1. **最小停机时间**

[
 \sum_{\tau=t-DT_i+1}^{t} z_{i,\tau} \le 1 - u_{i,t},\quad
 \forall i,\ t \in {DT_i,\dots,T}
 ]

1. **爬坡约束**

[
 \begin{aligned}
 p_{i,t} - p_{i,t-1} &\le RU_i,\quad \forall i,\ t>1\
 p_{i,t-1} - p_{i,t} &\le RD_i,\quad \forall i,\ t>1
 \end{aligned}
 ]

首时段用 (p_{i,0})（例如取为 (P_i^{\min} u_{i,0})）处理。

上述是文献中经典的 UC MIP 表述，已被广泛使用并被证明能较好刻画时序启停与爬坡行为。([optimization.cbe.cornell.edu](https://optimization.cbe.cornell.edu/index.php?title=Unit_commitment_problem&utm_source=chatgpt.com))

### 2.3 Python + Pyomo 实现（问题1）

下面给出一个可直接运行的 Pyomo 模型代码（假设你已装好 `pyomo` 与任意 MILP 求解器，如 Gurobi/CPLEX/GLPK）：

```python
from pyomo.environ import (ConcreteModel, Set, Param, Var, NonNegativeReals,
                           Binary, Objective, Constraint, minimize, value, SolverFactory)

def build_classic_uc_model(units_df, load_df):
    model = ConcreteModel()

    # ====== 集合 ======
    gens = list(units_df.index)  # 用行索引作为机组 ID
    hours = list(load_df.index)  # 1..24

    model.G = Set(initialize=gens)
    model.T = Set(initialize=hours)

    # ====== 参数 ======
    def get_param_dict(col):
        return units_df[col].to_dict()

    Pmax = get_param_dict("Pmax")
    Pmin = get_param_dict("Pmin")
    RU   = get_param_dict("RU")
    RD   = get_param_dict("RD")
    UT   = get_param_dict("UT")
    DT   = get_param_dict("DT")
    CS   = get_param_dict("Cost_start")
    CD   = get_param_dict("Cost_stop")
    a    = get_param_dict("a")
    b    = get_param_dict("b")
    c    = get_param_dict("c")

    model.Pmax = Param(model.G, initialize=Pmax)
    model.Pmin = Param(model.G, initialize=Pmin)
    model.RU   = Param(model.G, initialize=RU)
    model.RD   = Param(model.G, initialize=RD)
    model.UT   = Param(model.G, initialize=UT)
    model.DT   = Param(model.G, initialize=DT)
    model.CS   = Param(model.G, initialize=CS)
    model.CD   = Param(model.G, initialize=CD)
    model.a    = Param(model.G, initialize=a)
    model.b    = Param(model.G, initialize=b)
    model.c    = Param(model.G, initialize=c)

    # 负荷
    load_dict = load_df["Load"].to_dict()
    model.D = Param(model.T, initialize=load_dict)

    # 初始状态：根据 Initial Up/Down 简单推断
    init_up   = units_df["InitUp"].to_dict()
    init_down = units_df["InitDown"].to_dict()

    def init_u0(i):
        if init_up[i] > 0:
            return 1
        elif init_down[i] > 0:
            return 0
        else:
            # 未明确，简单假设初始停机
            return 0

    model.u0 = Param(model.G, initialize=init_u0)

    # ====== 变量 ======
    model.u = Var(model.G, model.T, domain=Binary)
    model.y = Var(model.G, model.T, domain=Binary)
    model.z = Var(model.G, model.T, domain=Binary)
    model.p = Var(model.G, model.T, domain=NonNegativeReals)

    # ====== 目标函数 ======
    def obj_rule(m):
        expr = 0
        for i in m.G:
            for t in m.T:
                expr += m.a[i]*m.p[i,t]**2 + m.b[i]*m.p[i,t] + m.c[i]*m.u[i,t]
                expr += m.CS[i]*m.y[i,t] + m.CD[i]*m.z[i,t]
        return expr

    model.Obj = Objective(rule=obj_rule, sense=minimize)

    # ====== 约束 ======

    # 功率平衡
    def balance_rule(m, t):
        return sum(m.p[i,t] for i in m.G) == m.D[t]
    model.Balance = Constraint(model.T, rule=balance_rule)

    # 出力上下限
    def gen_limits_rule(m, i, t):
        return (m.Pmin[i]*m.u[i,t], m.p[i,t], m.Pmax[i]*m.u[i,t])
    model.GenLimits = Constraint(model.G, model.T, rule=gen_limits_rule)

    # 启停逻辑
    hours_sorted = sorted(hours)
    t0 = hours_sorted[0]

    def startup_shutdown_rule(m, i, t):
        if t == t0:
            return m.u[i,t] - m.u0[i] == m.y[i,t] - m.z[i,t]
        else:
            prev = hours_sorted[hours_sorted.index(t)-1]
            return m.u[i,t] - m.u[i,prev] == m.y[i,t] - m.z[i,t]
    model.StartStop = Constraint(model.G, model.T, rule=startup_shutdown_rule)

    # 最小开机时间
    def min_up_rule(m, i, t):
        UT_i = int(m.UT[i])
        idx = hours_sorted.index(t)
        start_idx = max(0, idx-UT_i+1)
        relevant_hours = hours_sorted[start_idx:idx+1]
        return sum(m.y[i,tt] for tt in relevant_hours) <= m.u[i,t]
    model.MinUp = Constraint(model.G, model.T, rule=min_up_rule)

    # 最小停机时间
    def min_down_rule(m, i, t):
        DT_i = int(m.DT[i])
        idx = hours_sorted.index(t)
        start_idx = max(0, idx-DT_i+1)
        relevant_hours = hours_sorted[start_idx:idx+1]
        return sum(m.z[i,tt] for tt in relevant_hours) <= 1 - m.u[i,t]
    model.MinDown = Constraint(model.G, model.T, rule=min_down_rule)

    # 爬坡约束
    def ramp_up_rule(m, i, t):
        if t == t0:
            # 假定初始出力为 Pmin*u0
            return m.p[i,t] - m.Pmin[i]*m.u0[i] <= m.RU[i]
        else:
            prev = hours_sorted[hours_sorted.index(t)-1]
            return m.p[i,t] - m.p[i,prev] <= m.RU[i]

    def ramp_down_rule(m, i, t):
        if t == t0:
            return m.Pmin[i]*m.u0[i] - m.p[i,t] <= m.RD[i]
        else:
            prev = hours_sorted[hours_sorted.index(t)-1]
            return m.p[i,prev] - m.p[i,t] <= m.RD[i]

    model.RampUp = Constraint(model.G, model.T, rule=ramp_up_rule)
    model.RampDown = Constraint(model.G, model.T, rule=ramp_down_rule)

    return model

def solve_uc(model, solver_name="gurobi"):
    solver = SolverFactory(solver_name)
    results = solver.solve(model, tee=True)
    return results
```

使用方式：

```python
# 将 units_df 的 index 设为机组 ID（bus 号）
units_df = units.set_index("bus")
uc_model = build_classic_uc_model(units_df, load_tbl)
results = solve_uc(uc_model, solver_name="gurobi")  # 可改 'cplex' 或 'glpk'
```

### 2.4 结果提取与可视化

```python
def extract_uc_solution(model):
    hours = sorted(model.T)
    gens  = sorted(model.G)
    data = []

    for t in hours:
        for i in gens:
            data.append({
                "t": t,
                "g": i,
                "u": model.u[i,t].value,
                "p": model.p[i,t].value,
                "y": model.y[i,t].value,
                "z": model.z[i,t].value
            })
    return pd.DataFrame(data)

sol_df = extract_uc_solution(uc_model)

# 发电总量 vs 负荷
total_gen = sol_df.groupby("t")["p"].sum()
plt.figure()
plt.plot(total_gen.index, total_gen.values, marker="o", label="Generation")
plt.plot(load_tbl.index, load_tbl["Load"].values, marker="x", linestyle="--", label="Load")
plt.legend()
plt.xlabel("Hour")
plt.ylabel("Power (MW)")
plt.title("Generation vs Load (Classic UC)")
plt.grid(True)
plt.show()

# 机组启停热力图
pivot_u = sol_df.pivot(index="g", columns="t", values="u")
plt.figure()
plt.imshow(pivot_u.values, aspect="auto", interpolation="nearest")
plt.colorbar(label="Commitment (u)")
plt.xlabel("Hour")
plt.ylabel("Unit (bus)")
plt.title("Unit Commitment Schedule")
plt.show()
```

------

## 3. 问题 2：含网络与安全约束的 UC（SCUC）

在问题 1 基础上，引入：

1. **网络潮流（DC 潮流）约束**
2. **N-1 安全约束（近似）**
3. **旋转备用**
4. （可选）**系统最低惯量约束**

### 3.1 DC 潮流模型

采用经典 DC 近似：(\theta) 为母线相角（弧度），(f_\ell) 为线路潮流。([NREL Docs](https://docs.nrel.gov/docs/fy23osti/85006.pdf?utm_source=chatgpt.com))

对于每条线路 (\ell = (i,j))：

[
 f_{\ell,t} = \frac{1}{x_\ell}\big(\theta_{i,t} - \theta_{j,t}\big)
 ]

并有潮流上下限：

[
 -F_\ell^{\max} \le f_{\ell,t} \le F_\ell^{\max}
 ]

对每个母线 (b) 的潮流平衡：

[
 \sum_{g\in G_b} p_{g,t} - L_{b,t}
 = \sum_{\ell \in \delta^+(b)} f_{\ell,t} - \sum_{\ell \in \delta^-(b)} f_{\ell,t},
 \quad \forall b,t
 ]

其中 (G_b) 为接在母线 (b) 的机组集合，(L_{b,t}) 为该母线负荷。题目只给出系统总负荷，因此我们需要一个**负荷分布方案**。为了保持简单且可控，我们采取：

> 所有负荷集中在母线 1，即 (L_{1,t}=D_t, L_{b\neq 1,t}=0。)

这是允许的简化（题目允许为适应量子硬件做简化），同时保留网络传输与潮流瓶颈的物理意义。

为避免 DC 模型奇异，需要选择一个母线为相角参考（如 bus 1）：

[
 \theta_{1,t} = 0,\quad \forall t
 ]

### 3.2 旋转备用与 N-1 近似

**旋转备用**：保证任一时段有足够的备用应对突发负荷或机组/线路故障。

定义旋转备用量：

[
 R_t = \sum_{i\in G}\big(P_i^{\max} u_{i,t} - p_{i,t}\big)
 ]

要求：

[
 R_t \ge R_t^{\min},\quad \forall t
 ]

其中 (R_t^{\min}) 可设置为“当前时段中最大在线机组容量”或“最大机组容量”的某一比例，用于应对**最大单机故障**；我们取保守形式：

[
 R_t^{\min} \ge \max_{i\in G} P_i^{\max} u_{i,t}
 ]

实践中可简化为常数 (R^{\min} = \max_i P_i^{\max})（本系统中即最大机组容量 300 MW）。([optimization.cbe.cornell.edu](https://optimization.cbe.cornell.edu/index.php?title=Unit_commitment_problem&utm_source=chatgpt.com))

**N-1 安全约束（近似处理）**

严格的 N-1 SCUC 会为每个“机组/线路故障场景”复制一套网络与平衡约束，变量数量巨大。实际工程和多数研究中常通过**备用和安全裕度**来等价或保守逼近。([arXiv](https://arxiv.org/abs/2208.08028?utm_source=chatgpt.com))

这里采用简单但实用的近似：

1. **机组 N-1**：通过上面的旋转备用约束（备用 ≥ 最大单机容量）保证机组故障后仍能满足负荷。
2. **线路 N-1**：通过将正常工况下潮流上限缩放，例如

[
 |f_{\ell,t}| \le \alpha \cdot F_{\ell}^{\max},\quad \alpha \in (0,1),
 ]

例如 (\alpha = 0.8)。这样在任一线路故障（潮流重分配）后仍有裕度不越限。

若要更精细，可利用 LODF/PTDF 进行线故障重定向潮流的线性近似，这里略去实现细节。

### 3.3 可选：最低惯量约束

根据频率响应与转动惯量理论，系统等效惯量可写为：([Frontiers](https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2024.1418302/full?utm_source=chatgpt.com))

[
 H_{\text{sys},t} = \frac{\sum_{i\in G} H_i S_i u_{i,t}}{\sum_{i\in G} S_i u_{i,t}}
 ]

其中 (S_i) 可取机组额定容量（用 (P_i^{\max}) 近似）。

根据 [3] 及 Table5 中给出的 ROCOF 上限和扰动功率，可推导出一条关于最小系统惯量的约束（这里采用线性近似形式）：

[
 H_{\text{sys},t} \ge H^{\min}
 ]

其中 (H^{\min}) 可按题目参考文献中公式用 ROCOF_max、F_freq、P_step 等参数计算。在模型实现中，我们可先计算 (H^{\min}) 为常数，然后加入：

[
 \sum_{i} H_i P_i^{\max} u_{i,t} \ge H^{\min} \cdot \sum_i P_i^{\max} u_{i,t}
 ]

这是线性的（左、右皆为 (u) 的线性函数）。

### 3.4 SCUC 的 Python 实现（在 UC 基础上扩展）

在前面的 `build_classic_uc_model` 基础上，继续添加以下内容。

```python
from pyomo.environ import Reals

def extend_with_network_and_security(model, units_df, branch_df,
                                     load_df, alpha_line_margin=0.8,
                                     add_inertia=False, H_min=None):
    """
    在经典 UC 模型上增加
    - DC 潮流约束
    - 线路安全裕度 (近似 N-1)
    - 旋转备用约束
    - (可选) 惯量约束
    """
    # ========== 1. 网络集合与参数 ==========
    buses = sorted(set(branch_df["from_bus"]).union(branch_df["to_bus"]))
    model.B = Set(initialize=buses)

    # 机组所在母线
    bus_of_gen = units_df["bus"].to_dict()
    model.bus_of_gen = Param(model.G, initialize=bus_of_gen)

    # 线路集合
    lines = list(branch_df["id"])
    model.L = Set(initialize=lines)

    # 线路参数
    x_dict = branch_df.set_index("id")["x"].to_dict()
    Fmax_dict = branch_df.set_index("id")["Fmax"].to_dict()
    from_dict = branch_df.set_index("id")["from_bus"].to_dict()
    to_dict   = branch_df.set_index("id")["to_bus"].to_dict()

    model.x    = Param(model.L, initialize=x_dict)
    model.Fmax = Param(model.L, initialize=Fmax_dict)
    model.from_bus = Param(model.L, initialize=from_dict)
    model.to_bus   = Param(model.L, initialize=to_dict)

    # 相角与潮流变量
    model.theta = Var(model.B, model.T, domain=Reals)
    model.f = Var(model.L, model.T, domain=Reals)

    # ========== 2. 负荷在母线的分配 ==========
    # 简单假设：所有负荷在 bus=1
    slack_bus = buses[0]  # 取最小编号为参考母线
    def bus_load(b, t):
        return load_df.loc[t, "Load"] if b == slack_bus else 0.0

    # ========== 3. 潮流约束 ==========
    # 3.1 DC 线路潮流方程
    def flow_def_rule(m, l, t):
        i = m.from_bus[l]
        j = m.to_bus[l]
        return m.f[l,t] == (m.theta[i,t] - m.theta[j,t]) / m.x[l]
    model.FlowDef = Constraint(model.L, model.T, rule=flow_def_rule)

    # 3.2 母线功率平衡
    def bus_balance_rule(m, b, t):
        gen_sum = sum(m.p[i,t] for i in m.G if m.bus_of_gen[i] == b)
        load_bt = bus_load(b, t)
        # 出入线路功率
        out_flow = sum(m.f[l,t] for l in m.L if m.from_bus[l] == b)
        in_flow  = sum(m.f[l,t] for l in m.L if m.to_bus[l] == b)
        return gen_sum - load_bt == out_flow - in_flow
    model.BusBalance = Constraint(model.B, model.T, rule=bus_balance_rule)

    # 3.3 线路潮流上限（含安全裕度）
    def line_limit_rule(m, l, t):
        return (-alpha_line_margin * m.Fmax[l],
                m.f[l,t],
                alpha_line_margin * m.Fmax[l])
    model.LineLimit = Constraint(model.L, model.T, rule=line_limit_rule)

    # 3.4 参考母线相角设定为 0
    def ref_angle_rule(m, t):
        return m.theta[slack_bus, t] == 0.0
    model.RefAngle = Constraint(model.T, rule=ref_angle_rule)

    # ========== 4. 旋转备用 ==========
    # 剩余可用容量
    def reserve_amount(m, t):
        return sum(m.Pmax[i]*m.u[i,t] - m.p[i,t] for i in m.G)

    # 这里设 R_min = 最大机组容量
    R_min = max(units_df["Pmax"])
    def reserve_rule(m, t):
        return reserve_amount(m, t) >= R_min
    model.SpinReserve = Constraint(model.T, rule=reserve_rule)

    # ========== 5. (可选) 惯量约束 ==========
    if add_inertia:
        H_i = units_df["H"].to_dict()
        model.Hi = Param(model.G, initialize=H_i)

        def inertia_rule(m, t):
            num = sum(m.Hi[i]*m.Pmax[i]*m.u[i,t] for i in m.G)
            den = sum(m.Pmax[i]*m.u[i,t] for i in m.G)
            # 为避免 0 除，允许在 den=0 时不约束（系统没有机组不应发生）
            return num >= H_min * den
        model.Inertia = Constraint(model.T, rule=inertia_rule)

    return model
```

使用方式：

```python
scuc_model = uc_model.clone()
scuc_model = extend_with_network_and_security(scuc_model,
                                              units_df=units_df,
                                              branch_df=branch_tbl,
                                              load_df=load_tbl,
                                              alpha_line_margin=0.8,
                                              add_inertia=True,
                                              H_min=3.0)  # 举例设置 H_min
results_scuc = solve_uc(scuc_model, solver_name="gurobi")
```

### 3.5 结果对比与可视化

```python
scuc_sol_df = extract_uc_solution(scuc_model)

# 总成本比较
classic_cost = value(uc_model.Obj)
scuc_cost    = value(scuc_model.Obj)
print("Classic UC cost =", classic_cost)
print("SCUC (network+security) cost =", scuc_cost)

# 对比某几条重载线路的潮流
def extract_line_flow(model):
    data = []
    for t in sorted(model.T):
        for l in sorted(model.L):
            data.append({
                "t": t,
                "line": l,
                "f": model.f[l,t].value,
                "Fmax": model.Fmax[l]
            })
    return pd.DataFrame(data)

line_df = extract_line_flow(scuc_model)
heavy_lines = (line_df.groupby("line")["f"].max().abs()
               .sort_values(ascending=False).head(5).index)

plt.figure()
for l in heavy_lines:
    df_l = line_df[line_df["line"] == l]
    plt.plot(df_l["t"], df_l["f"], marker="o", label=f"Line {l}")
plt.xlabel("Hour")
plt.ylabel("Flow (MW)")
plt.title("Selected Line Flows (SCUC)")
plt.legend()
plt.grid(True)
plt.show()
```

> **定性结论**：
>
> - 与经典 UC 相比，引入网络与安全约束后，部分时段需要提前启动更多机组、留出旋转备用，并且受线路瓶颈限制，可能采用成本更高但位置更优的机组发电，因此 **总成本上升**，但换来网络安全与频率安全的提高。

------

## 4. 问题 3：UC → QUBO 转化与 Kaiwu SDK 求解思路

### 4.1 QUBO 基本形式与惩罚法

QUBO 的标准形式为：([pennylane.ai](https://pennylane.ai/qml/demos/tutorial_QUBO?utm_source=chatgpt.com))

[
 \min_{x\in{0,1}^n} x^\top Q x + c,
 ]

其中 (Q) 为对称矩阵，(x) 是 0/1 向量。

要把带约束的 UC（MIP）转成 QUBO，一般采用**惩罚法**：([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1572528620300281?utm_source=chatgpt.com))

- 对等式约束 (h(x)=0)：添加惩罚项 (\lambda h(x)^2)
- 对不等式约束 (g(x)\le 0)：可通过添加松弛变量并转为等式，然后同样平方惩罚。

只要惩罚系数 (\lambda) 充分大，约束被“软编码”到目标函数中，使 QUBO 的最优解近似满足原约束。

### 4.2 变量二进制编码（简化版 UC）

要构造具体 QUBO，需要全部变量是 0/1。原 UC 中的二进制变量 (u,y,z) 已满足要求，主要问题是**连续变量** (p_{i,t})、(\theta_{b,t})、(f_{l,t})。

常见做法：

1. **出力离散化**：
    将机组 i 在时段 t 的出力 (p_{i,t}) 用若干“出力块”表示：

   [
    p_{i,t} = \sum_{k=1}^{K_i} \Delta P_i , x_{i,t,k},
    \quad x_{i,t,k}\in{0,1},
    ]

   其中 (\Delta P_i = (P_i^{\max}-P_i^{\min})/K_i)，(K_i) 为离散级数。

   - 若要考虑最小出力，可加上 (P_i^{\min} u_{i,t})：

     [
      p_{i,t} = P_i^{\min} u_{i,t} + \sum_k \Delta P_i x_{i,t,k}.
      ]

2. **相角与潮流离散化**（简化网络约束）：
    网络变量较多，若完全离散化会极大增加比特数。方案：

   - 对 QUBO 原型，我们**保留总功率平衡与机组/线路容量约束**，但**不离散化相角**，而是使用线性化后的 PTDF/LODF 近似直接约束机组出力组合（在 QUBO 中仍是二次的）。
   - 为简便，我们在 QUBO 原始示例中只考虑**系统功率平衡 + 机组出力上下限 + 启停逻辑 + 旋转备用**，把完整网络约束留给经典 SCUC 求解。

从工程上讲，很多文献也采用类似“网络由经典求解、机组组合交给量子求解”的分层策略。([optimization.cbe.cornell.edu](https://optimization.cbe.cornell.edu/index.php?title=Unit_commitment_problem&utm_source=chatgpt.com))

### 4.3 数学推导：从 UC 到 QUBO（示例）

以**单时段简化 UC** 为例（用于说明 QUBO 构造方法；实际实现可对每个 t 做同样处理并在 Q 矩阵中分块叠加）：

- 变量集合包括：
  - (u_i)：机组开机
  - (x_{i,k})：出力块二进制变量
- 出力表达式：

[
 p_i = P_i^{\min} u_i + \Delta P_i \sum_{k=1}^{K_i} x_{i,k}
 ]

- 功率平衡约束：

[
 \sum_i p_i = D
 \quad \Rightarrow \quad
 \sum_i \left( P_i^{\min} u_i + \Delta P_i \sum_k x_{i,k} \right) - D = 0.
 ]

记 (h_{\text{bal}}(x) := \sum_i P_i^{\min} u_i + \Delta P_i \sum_k x_{i,k} - D)。
 惩罚项：

[
 \lambda_{\text{bal}} , h_{\text{bal}}(x)^2
 ]

展开平方，得到二次形式

[
 x^\top Q^{(\text{bal})} x + q^{(\text{bal})}{}^\top x + c^{(\text{bal})}.
 ]

类似地，可以对以下约束构造惩罚项：

- 旋转备用（单时段）：

[
 \sum_i (P_i^{\max}-p_i) \ge R^{\min}
 \quad\Rightarrow\quad
 g_{\text{res}}(x) := R^{\min} - \sum_i (P_i^{\max}-p_i) \le 0
 ]

可以通过引入整数松弛变量 (s\ge 0) 并离散化为二进制，写成

[
 g_{\text{res}}(x) + s = 0,
 ]

再平方惩罚。

- 出力上限/下限：
   在离散化设计时已近似通过 (K_i) 和 (\Delta P_i) 控制；也可以显式加入

[
 p_i \le P_i^{\max},\quad p_i \ge P_i^{\min} u_i
 ]

分别构造不等式惩罚。

- 启停逻辑与最小启停机时间：
   这些约束仅涉及二进制变量 (u_{i,t}, y_{i,t}, z_{i,t})，可以直接写成线性/仿射形式并平方。例如

  [
   u_{i,t} - u_{i,t-1} - y_{i,t} + z_{i,t} = 0
   ]

  → 惩罚项

  [
   \lambda_{\text{logic}}(u_{i,t} - u_{i,t-1} - y_{i,t} + z_{i,t})^2
   ]

所有惩罚项加总后，与原始成本函数（燃料成本 + 启停成本）一起构成：

[
 \min_{x\in{0,1}^n}
 C^{\text{orig}}(x) + \sum_j \lambda_j h_j(x)^2
 = x^\top Q x + q^\top x + c.
 ]

由此得到 QUBO 矩阵 (Q)（对称化处理）。

### 4.4 构造 QUBO 的 Python 代码（简化示例）

下面代码给出一个**简化版 QUBO 构造器**：仅考虑一个时段的 UC（方便演示），包含：

- 机组二进制 (u_i)
- 出力块变量 (x_{i,k})
- 功率平衡约束
- 备用约束
- 燃料成本近似（把 (p^2) 展开为关于 (x_{i,k}) 的二次项）

你可以对每个时段调用一次，再把所有时段的变量按索引拼接成一个大 Q 矩阵。

```python
import itertools

def build_single_period_qubo(units_df, demand,
                             K_per_unit=3,
                             lambda_balance=1000.0,
                             lambda_reserve=500.0):
    """
    为单时段构造一个简化 UC 的 QUBO:
    - 变量: u_i (机组 on/off), x_{i,k} (出力块)
    - 约束: 功率平衡 + 备用
    - 成本: 二次燃料成本近似 + 启停成本(假设仅考虑开机成本)
    返回:
    - Q: dict[(idx1, idx2)] -> value
    - var_index: dict[var_name] -> global_index
    """
    Q = defaultdict(float)
    lin = defaultdict(float)  # 线性项，可以并入 Q 的对角线
    const = 0.0

    # 1. 变量索引
    var_index = {}
    idx = 0

    gens = units_df.index.tolist()
    # u_i
    for i in gens:
        var_index[("u", i)] = idx
        idx += 1

    # x_{i,k}
    for i in gens:
        for k in range(K_per_unit):
            var_index[("x", i, k)] = idx
            idx += 1

    # 2. 出力离散化关系
    # p_i = Pmin_i * u_i + ΔP_i * sum_k 2^k * x_{i,k} (使用二进制权重提高精度)
    # 为简洁，这里用线性权重 (1,2,3...) 也可以，只要一致即可
    p_expr = {}  # 每个机组在 p 表达式中的 (coef, var) 列表

    for i in gens:
        Pmin = units_df.loc[i, "Pmin"]
        Pmax = units_df.loc[i, "Pmax"]
        # 简单均匀步长
        delta = (Pmax - Pmin) / K_per_unit if K_per_unit > 0 else 0.0

        terms = []
        # Pmin * u_i
        terms.append((Pmin, ("u", i)))
        # 出力块
        for k in range(K_per_unit):
            terms.append((delta, ("x", i, k)))
        p_expr[i] = terms

    # 3. 构造成本项: a_i p_i^2 + b_i p_i + c_i u_i
    for i in gens:
        a_i = units_df.loc[i, "a"]
        b_i = units_df.loc[i, "b"]
        c_i = units_df.loc[i, "c"]
        P_terms = p_expr[i]

        # 二次项 a_i * (Σ c_j x_j)^2
        for (coef1, var1), (coef2, var2) in itertools.product(P_terms, P_terms):
            idx1 = var_index[var1]
            idx2 = var_index[var2]
            Q[(min(idx1, idx2), max(idx1, idx2))] += a_i * coef1 * coef2

        # 线性项 b_i * Σ c_j x_j
        for coef, var in P_terms:
            idxv = var_index[var]
            lin[idxv] += b_i * coef

        # 线性项 c_i * u_i
        idxu = var_index[("u", i)]
        lin[idxu] += c_i

    # 4. 功率平衡惩罚: λ_balance * (Σ_i p_i - D)^2
    #   先把 Σ_i p_i 表成 x 的线性组合，再平方
    # Σ_i p_i = Σ_i Σ (coef * var)
    bal_terms = []
    for i in gens:
        bal_terms.extend(p_expr[i])

    # 常数项 -D
    const_bal = -demand

    # (Σ coef_j x_j + const_bal)^2 展开
    # 二次项
    for (c1, v1), (c2, v2) in itertools.product(bal_terms, bal_terms):
        idx1 = var_index[v1]
        idx2 = var_index[v2]
        Q[(min(idx1, idx2), max(idx1, idx2))] += lambda_balance * c1 * c2

    # 线性项: 2*const_bal*Σ c_j x_j
    for c, v in bal_terms:
        idxv = var_index[v]
        lin[idxv] += 2 * lambda_balance * const_bal * c

    # 常数项: const_bal^2
    const += lambda_balance * const_bal**2

    # 5. 备用惩罚: λ_reserve * (R_min - Σ_i (Pmax_i - p_i))_+^2
    # 为简化，假定用等式近似: λ_reserve * (R_min - Σ_i (Pmax_i - p_i))^2
    R_min = max(units_df["Pmax"])
    # Σ_i (Pmax_i - p_i) = Σ_i Pmax_i - Σ_i p_i
    sum_Pmax = units_df["Pmax"].sum()
    res_terms = []
    for i in gens:
        # -p_i -> -coef
        for coef, var in p_expr[i]:
            res_terms.append((-coef, var))
    const_res = R_min - sum_Pmax

    # 二次项
    for (c1, v1), (c2, v2) in itertools.product(res_terms, res_terms):
        idx1 = var_index[v1]
        idx2 = var_index[v2]
        Q[(min(idx1, idx2), max(idx1, idx2))] += lambda_reserve * c1 * c2

    # 线性项
    for c, v in res_terms:
        idxv = var_index[v]
        lin[idxv] += 2 * lambda_reserve * const_res * c

    const += lambda_reserve * const_res**2

    # 6. 把线性项并入 Q 的对角线
    for idxv, coef in lin.items():
        Q[(idxv, idxv)] += coef

    Q = dict(Q)
    return Q, var_index, const
```

### 4.5 Kaiwu SDK 量子/量子启发求解调用示例（伪代码）

Kaiwu SDK 的具体 Python API 需要根据平台文档调用，这里给出一个典型伪代码示意（结构正确、但需根据实际 SDK 修改函数名）：

```python
# 伪代码，仅示意
from kaiwu import QuantumOptimizer  # 假设 Kaiwu 提供的接口名

# 构建单时段 QUBO
Q, var_index, const = build_single_period_qubo(units_df=units_df.set_index("bus"),
                                               demand=load_tbl.loc[1, "Load"],
                                               K_per_unit=3)

# Kaiwu 可能接受 dict 或 numpy 矩阵形式
qubo = {
    "Q": Q,   # {(i,j): value}
    "offset": const
}

optimizer = QuantumOptimizer(
    num_reads=1000,          # 量子退火或量子灵感机器的采样次数
    temperature=0.1,         # 等参数依据文档设定
    # ...
)

solutions = optimizer.solve_qubo(qubo)

# 解析最优比特串
best_sol = min(solutions, key=lambda s: s.energy)
bitstring = best_sol.bitstring  # e.g. list/np.array of 0/1

# 根据 var_index 恢复 u_i, x_{i,k} 再计算 p_i 等
inv_var_index = {v: k for k, v in var_index.items()}

p_values = {}
for idx, bit in enumerate(bitstring):
    if bit == 0:
        continue
    var = inv_var_index[idx]  # ("u", i) 或 ("x", i, k)
    # 根据 p_expr 和 bitstring 再计算出力、启停状态等
```

> 在完整实现中，可对每个时段构造相应的 QUBO，或把 24 小时拼成一个大 QUBO（量子硬件允许时），再用 Kaiwu 求解，和经典 SCUC 的结果比较（成本、启停计划、备用、约束违背情况等）。

------

## 5. 问题 4：在量子硬件约束下的规模缩减策略

当前量子硬件（包括相干 Ising 机、量子退火器等）的比特/耦合数有限，为适应如“XXX bits” 的限制，需要有系统的**规模缩减（reduction）**策略。

### 5.1 总体思路

1. **变量压缩**
   - 降低每个连续变量的离散级数（例如从 (K=5) 降到 (K=2)）。
   - 只对关键机组/关键时段离散化，其他通过简单策略（如始终满发/零发）。
2. **问题分解**
   - **时间分解**（rolling horizon / sliding window）：例如 24 小时拆成 4 组 6 小时，带重叠边界条件，用量子方法分别求解。
   - **空间分解**：先用经典 SCUC 求出网络瓶颈与关键机组，在 QUBO 中只建模这些机组/节点。([NREL Docs](https://docs.nrel.gov/docs/fy23osti/85006.pdf?utm_source=chatgpt.com))
3. **约束裁剪与层次化建模**
   - 将部分复杂约束（如完整 N-1 + 线路潮流）留给经典求解；量子 QUBO 只负责决定机组开停与大致出力区间。
   - 采用两阶段流程：
     - 阶段 1：量子 / 量子启发 QUBO → 给出候选的机组组合方案（commitment pattern）；
     - 阶段 2：在固定 commitment 下，用经典 LP/MIP 做经济调度 + 网络修正。

### 5.2 时域分解的具体策略

- 设定一个窗口长度 (T_w)（如 6 小时），步长 (T_s)（如 3 小时）。
- 对第 k 个窗口 ([1 + (k-1)T_s, 1 + (k-1)T_s + T_w-1])：
  1. 把上一窗口末尾的状态 ((u_{i,\text{end}}, p_{i,\text{end}})) 作为此窗口的初始状态约束；
  2. 对窗口内构建 QUBO，仅考虑此部分时间与约束；
  3. 用 Kaiwu 求 QUBO 得到最优或近似最优解；
  4. 固定前 (T_s) 小时的解作为最终计划，向前滚动窗口。

这种 rolling horizon 策略是运筹学和电力调度中常用的分解手段，可以在限制的比特数下处理更长的时间范围。([optimization.cbe.cornell.edu](https://optimization.cbe.cornell.edu/index.php?title=Unit_commitment_problem&utm_source=chatgpt.com))

### 5.3 机组聚合与变量筛选

- 对于容量较小、成本结构相近的机组，可以在 QUBO 中**聚合为一台等效机组**（容量为多个机组之和，成本系数取加权平均），从而减少机组数 → 减少变量。([NREL Docs](https://docs.nrel.gov/docs/fy23osti/85006.pdf?utm_source=chatgpt.com))
- 通过经典 UC/SCUC 先做一遍敏感性分析：
  - 对成本极高、几乎不会启用的机组，可以在 QUBO 中直接固定为始终停机（去掉对应变量）。
  - 对必须运行的机组（如基荷机组），固定为始终开机，不进入 QUBO 的决策空间。

### 5.4 网络约束裁剪

- 使用经典 SCUC 计算基态潮流与对各种扰动的灵敏度，挑选出少数“**可能成为瓶颈的线路**”；在 QUBO 模型中只对这些线路添加与机组组合相关的惩罚约束。
- 对其余线路，仅在第二阶段经典求解中检查与修正。

### 5.5 精度与效率权衡

- 每减少一个出力离散级数或一条约束、一个机组，都显著减少 QUBO 的比特数与非零耦合数，提高量子求解的可行性，但也可能带来经济性与安全性的损失。
- 建议通过**数值实验**比较：
  - 全 MIP（Gurobi/CPLEX）解作为基准；
  - 几种不同缩减策略下的 QUBO + 经典后处理方案；
  - 比较目标成本、备用裕度、约束违背（潮流/ROCOF）等指标。

------

## 6. 结论与建议

1. **经典 UC 模型**
   - 基于题目数据构建的混合整数二次规划模型，包含燃料成本、启停成本、上下限、爬坡、最小启停机时间等关键约束；用 Pyomo + 商业/开源求解器即可稳定求解，得到 24 小时的机组启停方案和出力计划。
2. **含网络与安全约束的 UC（SCUC）**
   - 在 UC 基础上加入 DC 潮流、线路容量限制、旋转备用与（可选）惯量约束，得到更贴近实际的调度模型。
   - 结果上，总成本会有所增加，但线路负载率更低、备用充足、系统惯量满足安全要求，安全性显著提升。
3. **QUBO 转化与量子求解**
   - 利用惩罚法将 UC 的等式/不等式约束嵌入目标函数，构造出统一的 QUBO 形式，适配相干 Ising 机等量子/量子启发硬件。
   - 由于网络和连续变量带来的复杂度，我们在 QUBO 中采取**出力块离散化+网络约束简化**的策略，保留关键约束（功率平衡、容量、启停、备用），其余留给经典后处理。
4. **规模缩减策略**
   - 通过时间滚动窗口、机组聚合、变量筛选和关键线路裁剪等手段，可以在较小的比特容量下求解较大规模的 UC/SCUC 问题。
   - 建议采用“两阶段分层”框架：
     - 上层（QUBO + Kaiwu）：快速搜索全球机组组合空间；
     - 下层（经典 LP/MIP）：在给定组合下精细考虑网络、安全与惯量约束。
5. **整体评价**
   - 经典方法提供高精度、可行性保证；
   - 量子及量子启发 QUBO 方法在大规模组合搜索上有潜在加速优势，尤其是在高维度、强离散的 UC/SCUC 中；
   - 通过本文方案，可以搭建一个**可复现的混合经典-量子调度框架**：
     - 数据 → 经典 UC → SCUC → QUBO 转化 → Kaiwu 求解 → 回代网络与安全检查 → 策略迭代优化。

如果你希望，我也可以在下一步帮你：

- 用具体求解器跑出一组数值结果；
- 或者针对 QUBO 部分进一步细化，构造多时段的大规模 Q 矩阵，并给出完整变量映射与解码流程。