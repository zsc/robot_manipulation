# 第14章：灵巧操作与双臂协调

## 章节概要

本章深入探讨机器人灵巧操作的核心技术，从单手的精细操作到双臂协调控制。我们将分析如何通过手内操作实现物体的精确控制，探讨灵巧手的运动学建模与接触力优化，介绍操作图在复杂任务规划中的应用，详细对比双臂系统的控制策略，深入研究柔顺装配的力控制技术，以及机器人工具使用的认知架构。这些技术构成了下一代智能机器人执行复杂操作任务的基础。

## 学习目标

- 掌握手内操作的运动学与动力学原理
- 理解灵巧手的接触建模与抓取稳定性分析
- 学会构建和使用操作图进行任务规划
- 掌握双臂协调控制的设计方法
- 理解柔顺装配中的力/位混合控制
- 了解机器人工具使用的认知模型

---

## 14.1 手内操作与重抓取策略

### 14.1.1 手内操作基础

手内操作(In-hand Manipulation)是指在不失去抓取的情况下，通过手指运动改变物体在手中的位姿。这种能力对于精细操作任务至关重要。

**操作分类**：
1. **滚动(Rolling)**：物体在指尖滚动，接触点连续变化
2. **滑动(Sliding)**：物体沿指尖滑动，保持相对姿态  
3. **枢轴旋转(Pivoting)**：物体绕某个接触点旋转
4. **指尖行走(Finger Gaiting)**：交替改变接触手指实现运动

**运动学约束**：

设物体位姿为 $\mathbf{x}_o \in SE(3)$，第 $i$ 个接触点的位置为 $\mathbf{p}_i$，接触法向为 $\mathbf{n}_i$，则滚动约束为：

$$\mathbf{v}_{o,i} + \boldsymbol{\omega}_o \times \mathbf{r}_i = \mathbf{J}_i(\mathbf{q}) \dot{\mathbf{q}}$$

其中 $\mathbf{r}_i = \mathbf{p}_i - \mathbf{x}_o$ 是接触点相对物体质心的位置。

### 14.1.2 接触模式切换

手内操作涉及离散的接触模式切换，需要混合系统建模：

```
状态空间 = {滚动, 滑动, 分离, 冲击}
```

**模式转换条件**：
- 滚动→滑动：切向力超过摩擦锥 $|\mathbf{f}_t| > \mu |\mathbf{f}_n|$
- 滑动→分离：法向力消失 $\mathbf{f}_n \leq 0$
- 分离→冲击：新接触建立

### 14.1.3 重抓取规划

重抓取(Regrasping)通过一系列抓取配置的切换来实现大范围的物体重定向。

**图搜索方法**：
1. 构建抓取图 $G = (V, E)$
   - 节点 $v_i$：稳定抓取配置
   - 边 $e_{ij}$：从配置 $i$ 到 $j$ 的转换
2. 使用 A* 或 RRT* 搜索最优路径

**转换可行性判断**：
```
可行转换需满足：
1. 力闭合维持
2. 无碰撞路径存在
3. 关节限位满足
```

### 14.1.4 手指协调控制

**虚拟物体框架**：

将物体动力学与手指动力学耦合：

$$\begin{bmatrix} \mathbf{M}_o & 0 \\ 0 & \mathbf{M}_h \end{bmatrix} \begin{bmatrix} \ddot{\mathbf{x}}_o \\ \ddot{\mathbf{q}} \end{bmatrix} + \begin{bmatrix} \mathbf{c}_o \\ \mathbf{c}_h \end{bmatrix} = \begin{bmatrix} \mathbf{G}^T \\ -\mathbf{J}^T \end{bmatrix} \mathbf{f}_c + \begin{bmatrix} 0 \\ \boldsymbol{\tau} \end{bmatrix}$$

其中 $\mathbf{G}$ 是抓取矩阵，$\mathbf{f}_c$ 是接触力。

---

## 14.2 灵巧手运动学与接触建模

### 14.2.1 多指手运动学

现代灵巧手通常采用拟人化设计，如Shadow Hand(20 DOF)、Allegro Hand(16 DOF)。

**指链运动学**：

每个手指可视为串联机械臂，采用DH参数或PoE表示。对于第 $i$ 个手指：

$$\mathbf{T}_i = \prod_{j=1}^{n_i} \exp(\hat{\boldsymbol{\xi}}_{ij} \theta_{ij}) \mathbf{T}_{i,0}$$

**工作空间分析**：

灵巧工作空间 $W_d$ 定义为所有手指都能到达的空间：

$$W_d = \bigcap_{i=1}^{n_f} W_i$$

可达工作空间 $W_r$ 为至少一个手指能到达的空间：

$$W_r = \bigcup_{i=1}^{n_f} W_i$$

### 14.2.2 接触力学建模

**点接触模型分类**：

1. **无摩擦点接触(PwoF)**：1个约束，只传递法向力
2. **硬手指接触(HF)**：3个约束，传递3D力
3. **软手指接触(SF)**：4个约束，传递3D力+绕法向力矩

**接触约束方程**：

对于软手指模型，单个接触的扳手空间：

$$\mathbf{W}_i = \begin{bmatrix} \mathbf{I}_3 & 0 \\ [\mathbf{r}_i]_\times & \mathbf{n}_i \end{bmatrix}$$

其中 $[\mathbf{r}_i]_\times$ 是反对称矩阵。

### 14.2.3 抓取稳定性分析

**力闭合条件**：

抓取矩阵 $\mathbf{G} = [\mathbf{W}_1, \mathbf{W}_2, ..., \mathbf{W}_k]$ 满足：

$$\text{rank}(\mathbf{G}) = 6 \quad \text{且} \quad \exists \mathbf{f}_c > 0: \mathbf{G}\mathbf{f}_c = 0$$

**形闭合判定**：

使用凸包测试：原点在接触扳手凸包内部。

$$\mathbf{0} \in \text{ConvexHull}(\{\pm\mathbf{w}_1, \pm\mathbf{w}_2, ..., \pm\mathbf{w}_k\})$$

### 14.2.4 接触力优化

**最小内力优化**：

$$\begin{align}
\min_{\mathbf{f}_c} \quad & ||\mathbf{f}_c||^2 \\
\text{s.t.} \quad & \mathbf{G}\mathbf{f}_c = \mathbf{w}_{ext} \\
& \mathbf{f}_{n,i} \geq f_{min} \\
& ||\mathbf{f}_{t,i}|| \leq \mu_i \mathbf{f}_{n,i}
\end{align}$$

可转化为二次锥规划(SOCP)求解。

---

## 14.3 操作图与任务规划

### 14.3.1 操作图表示

操作图(Manipulation Graph)是一种离散化的任务空间表示，用于复杂操作序列规划。

**图结构定义**：

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{C})$$

- $\mathcal{S}$：状态空间(物体位姿×抓取配置)
- $\mathcal{A}$：动作空间(抓取、放置、推动等)
- $\mathcal{T}$：状态转移函数
- $\mathcal{C}$：成本函数

### 14.3.2 任务分解与原语

**操作原语库**：

```
基础原语 = {
    Pick(obj, grasp_pose)
    Place(obj, target_pose)  
    Push(obj, direction, distance)
    Rotate(obj, axis, angle)
    Insert(obj1, obj2, tolerance)
}
```

**任务分解示例**：

装配任务 → [拾取零件A, 对准孔位, 插入, 拾取零件B, 装配B到A]

### 14.3.3 符号规划与几何推理

**PDDL表示**：

```pddl
(:action pick
  :parameters (?obj - object ?grasp - grasp_pose)
  :precondition (and 
    (clear ?obj)
    (hand-empty)
    (stable-grasp ?obj ?grasp))
  :effect (and
    (holding ?obj)
    (not (hand-empty))
    (not (at-pose ?obj))))
```

**几何可行性验证**：

每个符号动作需要底层几何验证：
- 逆运动学可解性
- 无碰撞路径存在性
- 抓取稳定性

### 14.3.4 在线重规划

**执行监控与异常处理**：

```
监控指标 = {
    位姿偏差: ||x_actual - x_planned||
    力偏差: ||f_actual - f_expected||
    时间超限: t > t_max
}
```

触发重规划条件：
1. 监控指标超阈值
2. 意外碰撞检测
3. 抓取失败

---

## 14.4 双臂协调：主从控制vs协同控制

### 14.4.1 双臂系统架构

**运动学耦合**：

双臂系统总自由度 = 左臂DOF + 右臂DOF + 基座DOF

对于共同操作物体：

$$\begin{bmatrix} \mathbf{v}_L \\ \mathbf{v}_R \end{bmatrix} = \begin{bmatrix} \mathbf{J}_L & 0 \\ 0 & \mathbf{J}_R \end{bmatrix} \begin{bmatrix} \dot{\mathbf{q}}_L \\ \dot{\mathbf{q}}_R \end{bmatrix}$$

约束条件：$\mathbf{v}_L = \mathbf{v}_R + \boldsymbol{\omega} \times \mathbf{r}_{LR}$

### 14.4.2 主从控制模式

**位置主从控制**：

```
从臂目标 = 变换(主臂位姿, 相对关系)
```

$$\mathbf{x}_{slave} = \mathbf{T}_{rel} \cdot \mathbf{x}_{master}$$

**力反馈双向控制**：

$$\begin{align}
\boldsymbol{\tau}_{master} &= \mathbf{J}_m^T(\mathbf{K}_p \Delta\mathbf{x} + \mathbf{K}_f \mathbf{f}_{slave}) \\
\boldsymbol{\tau}_{slave} &= \mathbf{J}_s^T(\mathbf{K}_p \Delta\mathbf{x} - \mathbf{K}_f \mathbf{f}_{master})
\end{align}$$

### 14.4.3 协同控制策略

**虚拟物体控制**：

将双臂看作操作同一虚拟物体：

$$\begin{bmatrix} \mathbf{f}_L \\ \mathbf{f}_R \end{bmatrix} = \begin{bmatrix} \mathbf{G}_L^T \\ \mathbf{G}_R^T \end{bmatrix}^+ \mathbf{w}_{desired} + \mathbf{N} \boldsymbol{\lambda}$$

其中 $\mathbf{N}$ 是零空间投影，$\boldsymbol{\lambda}$ 是内力。

**负载分配优化**：

$$\begin{align}
\min \quad & \alpha ||\mathbf{f}_L||^2 + (1-\alpha)||\mathbf{f}_R||^2 \\
\text{s.t.} \quad & \mathbf{f}_L + \mathbf{f}_R = \mathbf{f}_{total} \\
& \boldsymbol{\tau}_L \times \mathbf{f}_L + \boldsymbol{\tau}_R \times \mathbf{f}_R = \mathbf{M}_{total}
\end{align}$$

### 14.4.4 同步与协调

**时间同步**：

```
同步策略 = {
    硬同步: 等待最慢臂
    软同步: 时间窗口内协调
    异步: 独立执行+关键点同步
}
```

**碰撞避免**：

使用胶囊体(Capsule)近似，距离计算：

$$d_{ij} = ||\mathbf{p}_i - \mathbf{p}_j|| - (r_i + r_j)$$

排斥速度：

$$\mathbf{v}_{rep} = k_{rep} \cdot \max(0, d_{safe} - d_{ij}) \cdot \frac{\mathbf{p}_j - \mathbf{p}_i}{||\mathbf{p}_j - \mathbf{p}_i||}$$

---

## 14.5 柔顺装配与插孔任务

### 14.5.1 装配任务分析

**典型装配误差源**：
- 位置误差：±0.5-2mm
- 姿态误差：±1-3°
- 零件公差：±0.01-0.1mm
- 传感器噪声：±0.1-0.5mm

**接触状态分类**：

```
插孔状态 = {
    无接触
    单点接触
    线接触
    双点接触
    楔入(Wedging)
    卡阻(Jamming)
    成功插入
}
```

### 14.5.2 被动柔顺机构

**远程柔顺中心(RCC)**：

设计柔顺中心位于插入轴线上，使接触力自动产生对准力矩：

$$\mathbf{M} = \mathbf{K}_\theta (\boldsymbol{\theta}_d - \boldsymbol{\theta}) = \mathbf{r} \times \mathbf{f}_c$$

弹性矩阵设计：

$$\mathbf{K} = \begin{bmatrix} k_x & 0 & 0 & 0 & 0 & 0 \\ 0 & k_y & 0 & 0 & 0 & 0 \\ 0 & 0 & k_z & 0 & 0 & 0 \\ 0 & 0 & 0 & k_{\theta x} & 0 & 0 \\ 0 & 0 & 0 & 0 & k_{\theta y} & 0 \\ 0 & 0 & 0 & 0 & 0 & k_{\theta z} \end{bmatrix}$$

### 14.5.3 主动柔顺控制

**混合力/位控制**：

选择矩阵 $\mathbf{S}$ 定义控制模式：

$$\boldsymbol{\tau} = \mathbf{J}^T[\mathbf{S}\mathbf{f}_d + (\mathbf{I}-\mathbf{S})\mathbf{K}_p(\mathbf{x}_d - \mathbf{x})]$$

插孔任务的典型选择矩阵：

$$\mathbf{S} = \text{diag}(0, 0, 1, 1, 1, 0)$$

(z方向力控制，xy方向位置控制)

### 14.5.4 搜索策略

**螺旋搜索**：

$$\begin{align}
x(t) &= r(t)\cos(\omega t) \\
y(t) &= r(t)\sin(\omega t) \\
r(t) &= r_0 + v_r t
\end{align}$$

**力引导搜索**：

根据接触力方向调整搜索方向：

$$\mathbf{v}_{search} = k_s \cdot \text{normalize}(\mathbf{f}_{lateral})$$

**概率搜索(以不确定性椭球)**：

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{3/2}|\boldsymbol{\Sigma}|^{1/2}} \exp(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}))$$

---

## 14.6 工具使用与功能推理

### 14.6.1 工具表示与分类

**功能分类**：

```
工具类别 = {
    杠杆类: 撬棍、钳子
    切割类: 刀、剪刀
    旋转类: 扳手、螺丝刀
    打击类: 锤子
    容器类: 勺子、杯子
}
```

**几何-功能映射**：

工具功能由几何特征决定：
- 作用面(Active Surface)
- 抓握面(Grasp Surface)
- 功能轴(Functional Axis)

### 14.6.2 工具使用运动生成

**任务空间分解**：

$$\mathbf{x}_{tool} = \mathbf{x}_{hand} \oplus \mathbf{T}_{hand}^{tool}$$

工具约束传播：

$$\mathbf{J}_{task} = \mathbf{J}_{tool} \cdot \mathbf{Ad}_{T} \cdot \mathbf{J}_{hand}$$

其中 $\mathbf{Ad}_{T}$ 是伴随变换。

### 14.6.3 功能推理与迁移

**功能相似度度量**：

$$sim(t_1, t_2) = w_g \cdot sim_{geom} + w_k \cdot sim_{kine} + w_d \cdot sim_{dyn}$$

**零样本工具使用**：

基于功能分解的迁移学习：
1. 识别工具功能部件
2. 匹配已知工具模板
3. 迁移操作策略
4. 根据具体几何调整

### 14.6.4 学习型工具使用

**强化学习框架**：

状态空间：$s = [\mathbf{x}_{obj}, \mathbf{x}_{tool}, \mathbf{f}_{contact}]$
动作空间：$a = [\mathbf{v}_{tool}, \mathbf{f}_{applied}]$
奖励函数：$r = r_{task} + r_{efficiency} - r_{damage}$

**示范学习方法**：

从人类演示中学习工具使用：
1. 轨迹分割与原语提取
2. 接触模式识别
3. 力profile学习
4. 泛化到新工具

---

## 案例研究：OpenAI灵巧手魔方操作

### 背景与挑战

OpenAI在2019年展示了仅使用视觉和本体感知的机器人手解魔方，这是灵巧操作的里程碑。

**硬件配置**：
- Shadow Dexterous Hand: 24 DOF, 20个电机
- 3个RGB相机用于魔方状态估计
- 触觉传感：92个触觉传感器
- 控制频率：10Hz视觉，100Hz控制

### 技术方案

**分层控制架构**：

```
高层: 魔方求解器(Kociemba算法)
     ↓
中层: 操作原语规划器
     ↓  
低层: 强化学习控制器
```

**强化学习设置**：

- **状态空间**：指尖位置(60D) + 魔方位姿(7D) + 目标面(6D)
- **动作空间**：20个关节目标位置
- **奖励设计**：
  $$r = r_{rotation} + r_{stability} - r_{drop} - r_{time}$$

**域随机化(Domain Randomization)**：

```python
随机化参数 = {
    '魔方大小': [5.5cm, 6.5cm],
    '质量': [80g, 120g],
    '摩擦系数': [0.5, 1.5],
    '执行器延迟': [0ms, 40ms],
    '传感器噪声': N(0, σ²)
}
```

### 关键创新

1. **自动课程学习(ADR)**：
   根据性能自动调整随机化范围：
   $$\Delta_{range} = \alpha \cdot \text{sign}(performance - threshold)$$

2. **状态估计器**：
   使用CNN从3个视角估计魔方状态，准确率>95%

3. **故障恢复**：
   检测掉落并自动拾取重试

### 实验结果

- 成功率：解2x2魔方60%，解3x3魔方20%
- 平均操作时间：3分钟(2x2)，7分钟(3x3)
- 最大连续旋转：50次不掉落
- Sim2Real性能保持：~80%

### 工程启示

1. **仿真的关键作用**：使用MuJoCo训练了13,000年的仿真时间
2. **鲁棒性通过随机化**：极端的域随机化是成功的关键
3. **视觉反馈的重要性**：纯触觉难以准确估计魔方状态
4. **分层分解复杂任务**：将困难问题分解为可管理的子问题

---

## 本章小结

本章系统介绍了灵巧操作与双臂协调的核心技术：

1. **手内操作**提供了不依赖外部支撑改变物体位姿的能力，通过接触模式切换和重抓取规划实现复杂操作。

2. **灵巧手建模**需要考虑多指协调、接触力学和抓取稳定性，软手指模型能更准确描述实际接触。

3. **操作图**将连续操作空间离散化，通过符号规划与几何推理结合实现任务级规划。

4. **双臂协调**有主从和协同两种模式，需要处理运动学耦合、负载分配和碰撞避免。

5. **柔顺装配**通过被动RCC或主动力控制处理装配误差，搜索策略对成功率至关重要。

6. **工具使用**需要功能理解与运动迁移，是机器人智能的重要体现。

**关键公式回顾**：

- 手内操作约束：$\mathbf{v}_{o,i} + \boldsymbol{\omega}_o \times \mathbf{r}_i = \mathbf{J}_i \dot{\mathbf{q}}$
- 力闭合条件：$\text{rank}(\mathbf{G}) = 6$ 且 $\exists \mathbf{f}_c > 0: \mathbf{G}\mathbf{f}_c = 0$
- 混合力/位控制：$\boldsymbol{\tau} = \mathbf{J}^T[\mathbf{S}\mathbf{f}_d + (\mathbf{I}-\mathbf{S})\mathbf{K}_p(\mathbf{x}_d - \mathbf{x})]$

---

## 练习题

### 基础题

**14.1** 考虑一个三指灵巧手抓取球形物体。每个手指与球接触，形成硬手指接触(摩擦点接触)。请问：
a) 该抓取的约束维度是多少？
b) 如果要实现力闭合，接触点应如何分布？
c) 计算该配置的抓取矩阵维度。

<details>
<summary>答案</summary>

a) 每个硬手指接触提供3个约束(3D力)，总共9个约束维度。

b) 三个接触点应均匀分布在球的最大圆周上，相互间隔120°，这样能形成力闭合。

c) 抓取矩阵 $\mathbf{G} \in \mathbb{R}^{6 \times 9}$，6是物体扳手空间维度，9是接触力空间维度(3个接触×3维力)。
</details>

**Hint**: 考虑每个接触能传递的力的维度，以及物体的自由度。

**14.2** 在插孔装配任务中，孔直径10mm，轴直径9.9mm。如果位置误差服从正态分布 $\mathcal{N}(0, 0.5^2)$ mm，计算不使用搜索策略时一次插入的成功概率。

<details>
<summary>答案</summary>

间隙为0.05mm，成功插入需要位置误差小于0.05mm。

成功概率 = $P(|X| < 0.05) = 2\Phi(0.05/0.5) - 1 = 2\Phi(0.1) - 1$

其中 $\Phi$ 是标准正态分布函数。
$\Phi(0.1) \approx 0.5398$

成功概率 ≈ 0.0796 = 7.96%
</details>

**Hint**: 计算位置误差落在可接受范围内的概率。

**14.3** 双臂机器人协同搬运一个2kg的箱子。如果要求两臂均匀分担负载，每臂末端需要提供多大的垂直力？如果箱子质心偏离中心10cm，力矩如何分配？

<details>
<summary>答案</summary>

重力：$F_g = 2 \times 9.8 = 19.6$ N

均匀分担：每臂 $F_z = 9.8$ N

质心偏离产生力矩：$M = 0.1 \times 19.6 = 1.96$ Nm

假设两臂相距40cm，力矩由力差产生：
$\Delta F = M / 0.4 = 4.9$ N

一臂：14.7 N，另一臂：4.9 N
</details>

**Hint**: 应用静力平衡方程。

### 挑战题

**14.4** 设计一个手内操作控制器，实现圆柱体在三指手中绕其轴线旋转90度。考虑：
a) 如何协调三个手指的运动？
b) 如何保持抓取稳定性？
c) 如何处理滑动摩擦？

<details>
<summary>答案</summary>

a) 手指协调策略：
- 使用滚动接触，三指保持120°分布
- 指尖速度：$v_i = r\omega$，其中 $r$ 是圆柱半径，$\omega$ 是目标角速度
- 相位同步：三指运动相位差保持120°

b) 稳定性维持：
- 法向力控制：$F_n > mg/3\mu$ 保证不滑落
- 使用力闭合优化保持最小内力
- 实时监测接触力，动态调整

c) 滑动摩擦处理：
- 切向力约束：$|F_t| < \mu F_n$
- 使用粘滑(stick-slip)模型
- 通过增加法向力或降低速度避免滑动
</details>

**Hint**: 考虑滚动约束和摩擦锥约束的结合。

**14.5** 针对双臂协同装配任务，设计一个控制架构，其中一臂持有基座零件(位置控制)，另一臂执行插入(力控制)。如何处理两臂的耦合约束？

<details>
<summary>答案</summary>

控制架构设计：

1. 任务空间分解：
   - 基座臂：笛卡尔位置控制，保持稳定
   - 插入臂：混合力/位控制

2. 约束处理：
   - 相对位姿约束：$\mathbf{T}_{insert} = \mathbf{T}_{base} \cdot \mathbf{T}_{relative}$
   - 速度耦合：$\mathbf{v}_{insert} = \mathbf{v}_{base} + \mathbf{v}_{rel}$

3. 控制律：
   基座臂：$\boldsymbol{\tau}_1 = \mathbf{J}_1^T \mathbf{K}_p (\mathbf{x}_{1d} - \mathbf{x}_1)$
   
   插入臂：$\boldsymbol{\tau}_2 = \mathbf{J}_2^T[\mathbf{S}\mathbf{f}_d + (\mathbf{I}-\mathbf{S})\mathbf{K}_p(\mathbf{x}_{2d} - \mathbf{x}_2)]$
   
   其中 $\mathbf{S} = \text{diag}(0,0,1,1,1,0)$ 用于插入方向力控制

4. 协调策略：
   - 使用虚拟弹簧阻尼器连接两臂
   - 基座臂补偿插入反作用力
</details>

**Hint**: 考虑主从控制与阻抗控制的结合。

**14.6** 机器人需要使用锤子将钉子敲入木板。分析：
a) 锤子的最优抓握位置
b) 敲击轨迹规划
c) 冲击力控制策略

<details>
<summary>答案</summary>

a) 最优抓握位置：
- 位置选择：距锤头2/3处，平衡控制性和冲击效果
- 优化目标：最大化 $E_k = \frac{1}{2}I\omega^2$，其中 $I$ 是转动惯量
- 约束：抓握力足够抵抗冲击反作用力

b) 轨迹规划：
- 准备阶段：笛卡尔直线运动到起始位置
- 加速阶段：最大加速度约束下的时间最优轨迹
- 冲击阶段：保持速度直到接触
- 轨迹：$z(t) = z_0 - \frac{1}{2}at^2$，优化 $a$ 使末端速度最大

c) 冲击力控制：
- 检测接触：力传感器阈值触发
- 冲击后控制：快速切换到阻抗控制吸收反弹
- 力profile：目标冲击力 $F_{impact} = \sqrt{2mE_k}/\Delta t$
- 安全策略：限制最大冲击力防止损坏
</details>

**Hint**: 考虑动量守恒和能量传递效率。

**14.7** (开放性思考题) 未来的灵巧操作系统应该如何结合视觉、触觉和力觉信息？讨论多模态融合在提高操作成功率方面的潜力。

<details>
<summary>答案</summary>

多模态融合框架：

1. 分层融合架构：
   - 低层：传感器级融合(时间同步、空间配准)
   - 中层：特征级融合(互补信息提取)
   - 高层：决策级融合(不确定性推理)

2. 模态互补性：
   - 视觉：全局形状、位姿、遮挡预测
   - 触觉：局部几何、材质、滑动检测
   - 力觉：接触状态、稳定性、装配进度

3. 融合策略：
   - 贝叶斯融合处理不确定性
   - 注意力机制动态权重分配
   - 跨模态学习弥补传感器失效

4. 应用场景：
   - 透明/反光物体：触觉补充视觉
   - 遮挡情况：力觉引导盲操作
   - 精细装配：多模态验证成功

5. 未来发展：
   - 神经形态传感器实现更快响应
   - 学习型融合策略自适应调整
   - 端到端多模态策略学习
</details>

**Hint**: 思考每种模态的优势和局限性。

**14.8** 设计一个操作图来完成"组装宜家家具"任务。考虑需要哪些操作原语，如何处理错误恢复，以及如何优化装配序列。

<details>
<summary>答案</summary>

操作图设计：

1. 操作原语定义：
```
原语集 = {
  识别零件(part_id)
  拾取(part, grasp_type)
  对准(part1, part2, alignment_type)
  插入(part, hole, force_limit)
  旋转(part, angle, axis)
  拧紧(screw, torque_limit)
  翻转(assembly)
  验证(connection)
}
```

2. 图结构：
   - 节点：装配状态(已完成步骤集合)
   - 边：操作原语执行
   - 权重：时间成本 + 失败风险

3. 错误恢复机制：
   - 每个原语配备失败检测
   - 回退策略：返回最近稳定状态
   - 重试策略：参数微调后重试
   - 替代路径：预计算备选装配序列

4. 序列优化：
   - 约束：几何可达性、稳定性要求
   - 目标：最小化总时间和翻转次数
   - 使用A*搜索最优路径
   - 并行化独立子装配

5. 实施要点：
   - 视觉识别零件和预钻孔
   - 力控制确保配合
   - 双臂协同大部件
   - 在线进度监测
</details>

**Hint**: 考虑装配顺序的约束和并行化可能。

---

## 常见陷阱与错误

1. **抓取力过大导致的物体损坏**
   - 错误：盲目增加抓取力保证稳定
   - 正确：根据物体材质和任务需求优化最小力

2. **忽视接触模式转换**
   - 错误：假设接触始终保持滚动或滑动
   - 正确：实时监测并处理接触状态变化

3. **双臂碰撞**
   - 错误：独立规划两臂轨迹
   - 正确：联合规划或实时碰撞检测与避免

4. **装配卡阻**
   - 错误：增加力强行插入
   - 正确：检测卡阻，调整姿态重试

5. **工具滑落**
   - 错误：固定抓握力
   - 正确：根据工具使用动态调整

6. **力控制不稳定**
   - 错误：过高的力控制增益
   - 正确：根据环境刚度调整控制参数

---

## 最佳实践检查清单

### 灵巧操作设计审查

- [ ] 手指数量和自由度满足任务需求
- [ ] 接触传感器覆盖关键区域
- [ ] 抓取规划考虑了多种抓取模式
- [ ] 手内操作策略包含失败恢复
- [ ] 力控制回路稳定性经过验证

### 双臂协调审查

- [ ] 明确定义主从关系或协同策略
- [ ] 实施了碰撞检测和避免
- [ ] 负载分配策略合理
- [ ] 同步机制满足实时性要求
- [ ] 故障情况下的安全停止

### 装配任务审查

- [ ] 公差分析完成
- [ ] 搜索策略覆盖不确定范围
- [ ] 力/位混合控制模式正确
- [ ] 卡阻检测和恢复机制
- [ ] 成功验证方法可靠

### 工具使用审查

- [ ] 工具抓握位置优化
- [ ] 任务空间正确映射
- [ ] 冲击和振动处理
- [ ] 工具磨损监测
- [ ] 安全限位设置

### 系统集成审查

- [ ] 传感器数据同步
- [ ] 控制频率满足要求
- [ ] 计算资源分配合理
- [ ] 错误处理完善
- [ ] 性能指标可测量