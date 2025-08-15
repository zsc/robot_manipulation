# 第10章：阻抗控制与力控制

机器人与环境的物理交互是实现复杂操作任务的核心挑战。传统的位置控制在自由空间运动时表现优异，但在接触任务中往往力不从心——过高的位置增益可能导致过大的接触力，甚至损坏机器人或环境。本章系统介绍阻抗控制理论，这是一种优雅地统一位置与力控制的框架，通过调节机器人的动态行为来实现柔顺且稳定的物理交互。我们将深入探讨不同控制模式的数学基础、稳定性分析以及工程实现中的关键问题。

## 10.1 位置控制vs力控制vs阻抗控制

### 10.1.1 三种控制范式的基本原理

在机器人控制领域，存在三种基本的控制范式，每种都有其独特的优势和适用场景。

**位置控制（Position Control）** 是最直观的控制方式，控制器的目标是让机器人末端执行器达到期望位置 $\mathbf{x}_d$：

$$\boldsymbol{\tau} = \mathbf{K}_p(\mathbf{x}_d - \mathbf{x}) + \mathbf{K}_d(\dot{\mathbf{x}}_d - \dot{\mathbf{x}}) + \mathbf{g}(\mathbf{q})$$

其中 $\mathbf{K}_p$ 和 $\mathbf{K}_d$ 分别是位置和速度增益矩阵，$\mathbf{g}(\mathbf{q})$ 是重力补偿项。这种控制方式在自由空间运动中效果excellent，但在接触环境中会产生问题：微小的位置误差会导致巨大的接触力。

**力控制（Force Control）** 直接控制机器人与环境的接触力 $\mathbf{F}_d$：

$$\boldsymbol{\tau} = \mathbf{J}^T(\mathbf{F}_d + \mathbf{K}_f \int (\mathbf{F}_d - \mathbf{F}) dt)$$

其中 $\mathbf{J}$ 是雅可比矩阵，$\mathbf{K}_f$ 是力控制增益。纯力控制在接触任务中表现出色，但无法控制运动方向上的位置。

**阻抗控制（Impedance Control）** 则是一种更加灵活的方法，它不直接控制位置或力，而是控制位置与力之间的动态关系：

$$\mathbf{M}_d(\ddot{\mathbf{x}} - \ddot{\mathbf{x}}_d) + \mathbf{C}_d(\dot{\mathbf{x}} - \dot{\mathbf{x}}_d) + \mathbf{K}_d(\mathbf{x} - \mathbf{x}_d) = \mathbf{F}_{ext}$$

这里 $\mathbf{M}_d$、$\mathbf{C}_d$、$\mathbf{K}_d$ 分别是期望的惯性、阻尼和刚度矩阵，$\mathbf{F}_{ext}$ 是外部作用力。

### 10.1.2 适用场景对比

```
场景类型        位置控制    力控制    阻抗控制
--------------------------------------------
自由空间运动      优秀       差        良好
硬接触           差         优秀      良好
软接触           一般       良好      优秀
装配任务         差         一般      优秀
人机协作         危险       安全      最安全
计算复杂度       低         中        高
传感器需求       编码器     力传感器   两者皆需
```

### 10.1.3 数学模型与动力学耦合

考虑一个n自由度机械臂的动力学方程：

$$\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau} + \mathbf{J}^T\mathbf{F}_{ext}$$

在阻抗控制中，我们希望末端执行器表现出期望的阻抗特性。通过逆动力学控制，可以设计控制律：

$$\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})\mathbf{J}^{-1}[\ddot{\mathbf{x}}_d - \dot{\mathbf{J}}\dot{\mathbf{q}}] + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) - \mathbf{J}^T\mathbf{F}_{cmd}$$

其中 $\mathbf{F}_{cmd}$ 是根据阻抗控制律计算的虚拟力：

$$\mathbf{F}_{cmd} = \mathbf{M}_d^{-1}[\mathbf{K}_d(\mathbf{x}_d - \mathbf{x}) + \mathbf{C}_d(\dot{\mathbf{x}}_d - \dot{\mathbf{x}}) - \mathbf{F}_{ext}]$$

## 10.2 笛卡尔阻抗与关节阻抗

### 10.2.1 笛卡尔空间阻抗控制

笛卡尔阻抗控制直接在任务空间定义期望的动态行为。对于6维任务空间（3维位置+3维姿态），期望阻抗可以表示为：

$$\begin{bmatrix} \mathbf{F} \\ \boldsymbol{\tau}_e \end{bmatrix} = \begin{bmatrix} \mathbf{K}_p & \mathbf{0} \\ \mathbf{0} & \mathbf{K}_o \end{bmatrix} \begin{bmatrix} \Delta \mathbf{p} \\ \Delta \boldsymbol{\phi} \end{bmatrix} + \begin{bmatrix} \mathbf{D}_p & \mathbf{0} \\ \mathbf{0} & \mathbf{D}_o \end{bmatrix} \begin{bmatrix} \Delta \dot{\mathbf{p}} \\ \Delta \boldsymbol{\omega} \end{bmatrix}$$

其中 $\Delta \mathbf{p}$ 是位置误差，$\Delta \boldsymbol{\phi}$ 是姿态误差（可用轴角表示），$\mathbf{K}_p$、$\mathbf{K}_o$ 是位置和姿态刚度，$\mathbf{D}_p$、$\mathbf{D}_o$ 是相应的阻尼矩阵。

姿态误差的计算需要特别注意，常用的方法是通过四元数：

$$\Delta \boldsymbol{\phi} = 2 \cdot \text{vec}(\mathbf{q}_d \otimes \mathbf{q}^{-1})$$

其中 $\text{vec}(\cdot)$ 提取四元数的向量部分。

### 10.2.2 关节空间阻抗控制

关节阻抗控制在关节空间定义期望行为：

$$\boldsymbol{\tau} = \mathbf{K}_q(\mathbf{q}_d - \mathbf{q}) + \mathbf{D}_q(\dot{\mathbf{q}}_d - \dot{\mathbf{q}}) + \mathbf{g}(\mathbf{q})$$

关节阻抗与笛卡尔阻抗之间通过雅可比矩阵相关联：

$$\mathbf{K}_x = (\mathbf{J}^{-T} \mathbf{K}_q \mathbf{J}^{-1})|_{\mathbf{q}=\mathbf{q}_0}$$

这个关系表明，关节刚度通过运动学配置影响笛卡尔刚度，在奇异配置附近，笛卡尔刚度会显著降低。

### 10.2.3 可变刚度与零空间利用

对于冗余机器人（n > 6），可以利用零空间调节刚度椭球的形状而不改变末端位置：

$$\boldsymbol{\tau} = \mathbf{J}^T \mathbf{F}_{task} + (\mathbf{I} - \mathbf{J}^T \mathbf{J}^{T\dagger}) \boldsymbol{\tau}_{null}$$

其中 $\mathbf{J}^{T\dagger} = (\mathbf{J}\mathbf{J}^T)^{-1}\mathbf{J}$ 是雅可比的伪逆，$\boldsymbol{\tau}_{null}$ 是零空间力矩，可用于：
- 避免关节限位
- 优化可操作度
- 调节刚度椭球方向

## 10.3 接触稳定性与被动性

### 10.3.1 接触动力学建模

当机器人与环境接触时，系统动力学变为：

```
    机器人                  环境
      ↓                     ↓
M_r ẍ + C_r ẋ + K_r x = F_c
                          ↑
                    F_c = K_e(x - x_e) + C_e ẋ
```

闭环系统的稳定性取决于机器人阻抗 $Z_r(s)$ 与环境阻抗 $Z_e(s)$ 的相互作用：

$$Z_r(s) = M_r s^2 + C_r s + K_r$$
$$Z_e(s) = C_e s + K_e$$

### 10.3.2 稳定性条件

Hogan的稳定性准则指出，对于稳定的接触：
1. 机器人与环境中至少一个必须是被动的
2. 如果环境刚度很高，机器人应表现为低阻抗
3. 耦合端口的能量流必须满足被动性条件

稳定性可通过Nyquist图分析，避免 $Z_r(j\omega) + Z_e(j\omega) = 0$ 的情况。

### 10.3.3 被动性设计

为保证被动性，阻抗参数必须满足：

$$\mathbf{M}_d > 0, \quad \mathbf{K}_d \geq 0, \quad \mathbf{C}_d \geq 2\sqrt{\mathbf{M}_d \mathbf{K}_d}$$

最后一个条件确保系统是过阻尼或临界阻尼的。在实际实现中，通常选择：

$$\mathbf{C}_d = 2\zeta\sqrt{\mathbf{M}_d \mathbf{K}_d}$$

其中 $\zeta \in [0.7, 1.0]$ 是阻尼比。

## 10.4 混合力/位置控制架构

### 10.4.1 选择矩阵方法

Mason提出的选择矩阵 $\mathbf{S}$ 允许在不同方向上分别进行力控制和位置控制：

$$\boldsymbol{\tau} = \mathbf{J}^T[\mathbf{S}\mathbf{F}_d + (\mathbf{I} - \mathbf{S})\mathbf{F}_p]$$

其中：
- $\mathbf{F}_d$ 是期望力
- $\mathbf{F}_p = \mathbf{K}_p(\mathbf{x}_d - \mathbf{x}) + \mathbf{K}_d(\dot{\mathbf{x}}_d - \dot{\mathbf{x}})$ 是位置控制产生的虚拟力
- $\mathbf{S} = \text{diag}(s_1, s_2, ..., s_6)$，$s_i \in \{0, 1\}$

例如，对于平面抛光任务：
```
S = diag(1, 1, 0, 1, 1, 1)
        ↑  ↑  ↑  ↑  ↑  ↑
        x  y  z  α  β  γ
```
表示z方向力控制，其他方向位置控制。

### 10.4.2 约束框架

De Schutter和Bruyninckx提出的约束框架通过定义约束雅可比 $\mathbf{J}_c$ 来描述任务：

$$\mathbf{J}_c \dot{\mathbf{x}} = \mathbf{0}$$

控制律设计为：

$$\boldsymbol{\tau} = \mathbf{J}^T_c \boldsymbol{\lambda} + (\mathbf{I} - \mathbf{J}^T_c \mathbf{J}^{T\dagger}_c)\mathbf{J}^T \mathbf{F}_{motion}$$

其中 $\boldsymbol{\lambda}$ 是约束力，$\mathbf{F}_{motion}$ 控制约束允许的运动。

### 10.4.3 实现细节与切换策略

在力/位置控制模式切换时，需要注意：

1. **连续性保证**：切换瞬间的控制输出应连续
2. **积分器管理**：切换时重置积分项避免wind-up
3. **滤波器状态**：保持滤波器状态连续性

切换逻辑示例：
```
if (contact_detected && F_normal > F_threshold):
    S[2] = 0  # 切换到z方向力控制
    integral_force_z = 0  # 重置积分器
    F_d[2] = F_normal  # 设置期望力为当前接触力
```

## 10.5 案例研究：德国宇航中心DLR轻型机器人的软体控制

### 10.5.1 背景与挑战

德国宇航中心（DLR）的轻型机器人系列（LWR-III/IV、KUKA LBR iiwa）是阻抗控制领域的标杆。这些机器人专为安全人机协作设计，面临的核心挑战包括：

1. **高带宽力控制**：需要1kHz以上的控制频率
2. **关节弹性**：谐波减速器引入的弹性影响稳定性
3. **多模态控制**：位置、力、阻抗模式的无缝切换
4. **安全性保证**：碰撞检测与反应时间< 5ms

### 10.5.2 硬件架构

DLR机器人的独特设计：

```
关节传感器配置：
┌─────────────────────────┐
│  电机侧编码器（19位）      │ ← 电机位置 θ_m
├─────────────────────────┤
│  谐波减速器（100:1）      │ ← 弹性元件 K_joint
├─────────────────────────┤
│  连杆侧编码器（12位）      │ ← 连杆位置 θ_l
├─────────────────────────┤
│  关节力矩传感器           │ ← 力矩测量 τ_sensor
└─────────────────────────┘
```

双编码器设计允许直接测量关节弹性变形：
$$\tau_{elastic} = K_{joint}(\theta_m / N - \theta_l)$$

其中 $N$ 是减速比。

### 10.5.3 控制架构实现

DLR采用分层控制架构：

**第1层：关节力矩控制（3kHz）**
```
τ_cmd = τ_model + K_τ(τ_d - τ_sensor) + D_τ(τ̇_d - τ̇_sensor)
```

**第2层：笛卡尔阻抗控制（1kHz）**
```
F_d = M_d(ẍ_d - ẍ) + C_d(ẋ_d - ẋ) + K_d(x_d - x) + F_ext
τ_d = J^T F_d + null_space_torque
```

**第3层：任务规划（100Hz）**
- 轨迹生成
- 阻抗参数调度
- 模式切换逻辑

### 10.5.4 关键创新

1. **振动抑制**：通过状态反馈抑制弹性关节振动
   $$\tau_{motor} = \tau_{desired} + K_{damp}(\dot{\theta}_m - \dot{\theta}_l)$$

2. **碰撞检测**：基于动量观测器
   $$r = K_I \int (\tau_{ext} - \tau_{threshold}) dt$$
   当 $|r| > r_{max}$ 时触发安全反应。

3. **可变刚度控制**：根据任务阶段自适应调节
   ```
   装配接近阶段：K = 1000 N/m
   接触阶段：    K = 200 N/m
   插入阶段：    K = 50 N/m
   ```

### 10.5.5 性能指标

DLR LBR iiwa的典型性能：
- 位置精度：±0.1mm
- 力控制精度：±0.5N
- 最小可控阻抗：10 N/m
- 碰撞反应时间：< 4ms
- 安全接触力：< 50N（ISO 10218标准）

## 10.6 高级话题：分数阶控制与自适应阻抗

### 10.6.1 分数阶阻抗控制

传统阻抗控制使用整数阶微分，分数阶控制引入非整数阶微分算子：

$$M_d s^2 + C_d s^{\alpha} + K_d = Z_d(s), \quad 0 < \alpha < 2$$

分数阶微分的Caputo定义：
$$D^{\alpha} f(t) = \frac{1}{\Gamma(n-\alpha)} \int_0^t \frac{f^{(n)}(\tau)}{(t-\tau)^{\alpha-n+1}} d\tau$$

其中 $n-1 < \alpha < n$，$\Gamma(\cdot)$ 是伽马函数。

**优势**：
1. 更灵活的频率响应调节
2. 改善的鲁棒性
3. 更好的人类肌肉阻抗匹配（研究表明人类肌肉表现出分数阶特性）

**实现方法**：
Oustaloup近似将分数阶算子离散化：
$$s^{\alpha} \approx K \prod_{k=-N}^{N} \frac{s + \omega_k'}{s + \omega_k}$$

其中频率点按几何级数分布。

### 10.6.2 自适应阻抗控制

自适应阻抗根据环境特性和任务需求实时调节参数：

**基于力跟踪误差的自适应律**：
$$\dot{K}_d = -\gamma_K \mathbf{e}_F \mathbf{e}_x^T$$
$$\dot{C}_d = -\gamma_C \mathbf{e}_F \dot{\mathbf{e}}_x^T$$

其中 $\mathbf{e}_F = \mathbf{F}_d - \mathbf{F}$ 是力误差，$\mathbf{e}_x = \mathbf{x}_d - \mathbf{x}$ 是位置误差。

**基于强化学习的阻抗优化**：
使用策略梯度方法优化阻抗参数：
```
状态 s = [e_x, ė_x, F_ext, Ḟ_ext]
动作 a = [ΔK_d, ΔC_d, ΔM_d]
奖励 r = -w_1||e_F||² - w_2||e_x||² - w_3||a||²
```

### 10.6.3 变阻抗的生物启发

人类手臂阻抗调节机制：
1. **协同收缩**：同时激活拮抗肌增加刚度
2. **反射调节**：基于感觉反馈的快速调节
3. **预测性调节**：基于任务预期的前馈调节

仿生实现：
$$K_{arm} = K_{base} + \alpha \cdot co\_contraction + \beta \cdot reflex\_gain$$

## 10.7 本章小结

本章系统介绍了机器人阻抗控制的理论基础与工程实现。关键要点包括：

1. **控制模式选择**：位置控制适用于自由空间，力控制适用于约束空间，阻抗控制提供统一框架
2. **阻抗参数设计**：需考虑稳定性、被动性和任务需求
3. **笛卡尔vs关节空间**：笛卡尔空间直观但计算复杂，关节空间高效但缺乏物理直观性
4. **稳定性保证**：接触稳定性依赖于机器人与环境阻抗的匹配
5. **混合控制**：选择矩阵方法实现力/位置的解耦控制
6. **工程实现**：高性能阻抗控制需要优秀的硬件设计与多层控制架构

**核心公式汇总**：

阻抗控制基本方程：
$$\mathbf{M}_d(\ddot{\mathbf{x}} - \ddot{\mathbf{x}}_d) + \mathbf{C}_d(\dot{\mathbf{x}} - \dot{\mathbf{x}}_d) + \mathbf{K}_d(\mathbf{x} - \mathbf{x}_d) = \mathbf{F}_{ext}$$

稳定性条件：
$$\mathbf{C}_d \geq 2\sqrt{\mathbf{M}_d \mathbf{K}_d}$$

笛卡尔-关节刚度映射：
$$\mathbf{K}_x = \mathbf{J}^{-T} \mathbf{K}_q \mathbf{J}^{-1}$$

## 10.8 练习题

### 基础题

**练习10.1** 一个2自由度平面机械臂，连杆长度均为0.5m，当前配置为 $q_1 = 30°$，$q_2 = 60°$。如果关节刚度为 $\mathbf{K}_q = \text{diag}(100, 50)$ Nm/rad，计算末端笛卡尔刚度矩阵。

*Hint*: 先计算雅可比矩阵，然后应用刚度变换公式。

<details>
<summary>参考答案</summary>

雅可比矩阵：
$$\mathbf{J} = \begin{bmatrix} 
-0.5\sin(30°) - 0.5\sin(90°) & -0.5\sin(90°) \\
0.5\cos(30°) + 0.5\cos(90°) & 0.5\cos(90°)
\end{bmatrix} = \begin{bmatrix} 
-0.75 & -0.5 \\
0.433 & 0
\end{bmatrix}$$

笛卡尔刚度：
$$\mathbf{K}_x = \mathbf{J}^{-T} \mathbf{K}_q \mathbf{J}^{-1} \approx \begin{bmatrix} 
44.4 & 38.5 \\
38.5 & 83.3
\end{bmatrix}$$ N/m
</details>

**练习10.2** 设计一个阻抗控制器，使机器人末端在接触刚度为 $K_e = 10000$ N/m 的环境时保持稳定。期望接触力为10N，选择合适的阻抗参数。

*Hint*: 考虑阻抗比 $K_d/K_e$ 应该远小于1。

<details>
<summary>参考答案</summary>

为保证稳定性，选择：
- $K_d = 100$ N/m（$K_d/K_e = 0.01 << 1$）
- $M_d = 1$ kg
- $C_d = 2\sqrt{M_d K_d} = 20$ Ns/m（临界阻尼）

稳态接触力：$F = K_d \cdot \Delta x = 100 \times 0.1 = 10$ N
</details>

**练习10.3** 实现一个平面抛光任务的混合力/位置控制。机器人需要在xy平面内跟踪轨迹，同时在z方向保持5N的接触力。写出选择矩阵和控制律。

*Hint*: z方向力控制，xy方向位置控制。

<details>
<summary>参考答案</summary>

选择矩阵：
$$\mathbf{S} = \text{diag}(0, 0, 1, 0, 0, 0)$$

控制律：
$$\boldsymbol{\tau} = \mathbf{J}^T[\mathbf{S}\mathbf{F}_d + (\mathbf{I} - \mathbf{S})\mathbf{K}_p(\mathbf{x}_d - \mathbf{x})]$$

其中 $\mathbf{F}_d = [0, 0, 5, 0, 0, 0]^T$ N
</details>

### 挑战题

**练习10.4** 推导七自由度机械臂利用零空间调节末端刚度椭球方向的控制律。假设主任务是保持末端位置不变，次任务是最大化某个方向的刚度。

*Hint*: 使用梯度投影法在零空间优化刚度指标。

<details>
<summary>参考答案</summary>

定义刚度优化指标：
$$\phi = \mathbf{v}^T \mathbf{K}_x \mathbf{v}$$

其中 $\mathbf{v}$ 是期望的高刚度方向。

零空间控制律：
$$\boldsymbol{\tau}_{null} = k_0 (\mathbf{I} - \mathbf{J}^{\dagger}\mathbf{J}) \nabla_q \phi$$

梯度计算：
$$\nabla_q \phi = \frac{\partial}{\partial \mathbf{q}}(\mathbf{v}^T \mathbf{J}^{-T} \mathbf{K}_q \mathbf{J}^{-1} \mathbf{v})$$

完整控制律：
$$\boldsymbol{\tau} = \mathbf{J}^T \mathbf{F}_{task} + \boldsymbol{\tau}_{null}$$
</details>

**练习10.5** 分析分数阶阻抗 $C_d s^{1.5}$ 相比整数阶阻尼 $C_d s$ 的频率响应特性。绘制Bode图并讨论物理意义。

*Hint*: 计算 $|s^{1.5}|$ 和 $\angle s^{1.5}$ 在 $s = j\omega$ 时的值。

<details>
<summary>参考答案</summary>

分数阶项频率响应：
$$s^{1.5}|_{s=j\omega} = (j\omega)^{1.5} = \omega^{1.5} e^{j1.5\pi/2} = \omega^{1.5} e^{j135°}$$

幅值：$|\omega^{1.5}|$ = $\omega^{1.5}$（介于 $\omega$ 和 $\omega^2$ 之间）
相位：135°（介于90°和180°之间）

物理意义：
- 提供介于速度阻尼和加速度阻尼之间的特性
- 在高频具有更强的衰减（相比一阶）
- 在低频保持较好的响应（相比二阶）
- 更接近生物系统的阻尼特性
</details>

**练习10.6**（开放题）设计一个自适应阻抗控制器用于未知刚度表面的接触任务。控制器应能在线估计环境刚度并相应调整机器人阻抗。

*Hint*: 使用递归最小二乘(RLS)估计环境参数。

<details>
<summary>参考答案</summary>

环境模型：$F_{env} = K_e(x - x_e) + C_e \dot{x}$

RLS估计器：
$$\hat{\theta}_{k+1} = \hat{\theta}_k + \mathbf{P}_k \boldsymbol{\phi}_k (F_{measured} - \boldsymbol{\phi}_k^T \hat{\theta}_k)$$
$$\mathbf{P}_{k+1} = \frac{1}{\lambda}(\mathbf{P}_k - \frac{\mathbf{P}_k \boldsymbol{\phi}_k \boldsymbol{\phi}_k^T \mathbf{P}_k}{\lambda + \boldsymbol{\phi}_k^T \mathbf{P}_k \boldsymbol{\phi}_k})$$

其中 $\hat{\theta} = [K_e, C_e]^T$，$\boldsymbol{\phi} = [x-x_e, \dot{x}]^T$

自适应律：
$$K_d = \alpha / \hat{K}_e$$（保持阻抗比恒定）
$$C_d = 2\sqrt{M_d K_d}$$（维持临界阻尼）
</details>

**练习10.7** 证明在接触过程中，如果机器人阻抗 $Z_r(s)$ 和环境阻抗 $Z_e(s)$ 都是严格正实的(SPR)，则闭环系统是稳定的。

*Hint*: 使用Passivity定理和Lyapunov方法。

<details>
<summary>参考答案</summary>

定义存储函数：
$$V = \frac{1}{2}(M_r \dot{x}^2 + K_r x^2) + \frac{1}{2}K_e (x - x_e)^2$$

计算时间导数：
$$\dot{V} = \dot{x}(M_r \ddot{x} + K_r x) + K_e(x - x_e)\dot{x}$$

代入动力学方程：
$$M_r \ddot{x} + C_r \dot{x} + K_r x = -K_e(x - x_e) - C_e \dot{x}$$

得到：
$$\dot{V} = -C_r \dot{x}^2 - C_e \dot{x}^2 \leq 0$$

由于 $C_r, C_e > 0$（SPR条件），系统是稳定的。
</details>

**练习10.8** 为双臂协调操作设计阻抗控制策略。两个机械臂需要协同搬运一个刚性物体，如何分配内力和运动控制？

*Hint*: 定义虚拟刚度连接两臂，使用主从或对称控制架构。

<details>
<summary>参考答案</summary>

定义相对坐标：
- 物体位姿：$\mathbf{x}_{obj} = (\mathbf{x}_1 + \mathbf{x}_2)/2$
- 内力坐标：$\mathbf{x}_{int} = \mathbf{x}_1 - \mathbf{x}_2$

控制律：
臂1：$$\mathbf{F}_1 = \mathbf{K}_{obj}(\mathbf{x}_{obj,d} - \mathbf{x}_{obj}) + \mathbf{K}_{int}(\mathbf{x}_{int,d} - \mathbf{x}_{int})/2$$
臂2：$$\mathbf{F}_2 = \mathbf{K}_{obj}(\mathbf{x}_{obj,d} - \mathbf{x}_{obj}) - \mathbf{K}_{int}(\mathbf{x}_{int,d} - \mathbf{x}_{int})/2$$

参数选择：
- $\mathbf{K}_{obj}$：控制物体运动（高刚度）
- $\mathbf{K}_{int}$：控制内力（低刚度，避免过大内力）
- $\mathbf{x}_{int,d}$：期望抓取距离
</details>

## 10.9 常见陷阱与错误（Gotchas）

### 控制器设计陷阱

1. **刚度矩阵非对称**
   ```
   错误：K_d = [1000, 500; 200, 800]  # 非对称！
   正确：K_d = [1000, 350; 350, 800]  # 对称正定
   ```

2. **忽视采样时间影响**
   - 离散化破坏被动性
   - 解决：使用能量守恒的离散化方法

3. **雅可比奇异性**
   ```
   if (det(J*J') < 0.001):  # 接近奇异
       use damped_inverse  # 使用阻尼逆
   ```

4. **力传感器噪声**
   - 高增益 + 噪声 = 振荡
   - 解决：低通滤波，但注意相位延迟

### 实现陷阱

5. **坐标系不一致**
   - 力传感器坐标系 ≠ 机器人基座标系
   - 必须正确变换：$\mathbf{F}_{base} = \mathbf{R}_{sensor}^{base} \mathbf{F}_{sensor}$

6. **单位不匹配**
   ```
   K_joint = 100 Nm/rad  # 关节空间
   K_cart = 100 N/m      # 笛卡尔空间
   # 不能直接比较！
   ```

7. **积分饱和（Wind-up）**
   - 长时间接触导致积分项过大
   - 解决：anti-windup机制

8. **模式切换抖振**
   - 在力/位置控制边界频繁切换
   - 解决：滞回区或平滑过渡

### 调试技巧

9. **振荡诊断**
   ```
   观察频率：
   - 高频(>100Hz): 降低增益或增加阻尼
   - 低频(<10Hz): 检查积分项或机械谐振
   - 极限环: 检查非线性（摩擦、饱和）
   ```

10. **性能调优顺序**
    1. 先调重力补偿
    2. 纯位置控制
    3. 加入阻尼项
    4. 最后调节刚度

## 10.10 最佳实践检查清单

### 设计审查要点

#### 硬件配置
- [ ] 力/扭矩传感器安装位置优化（靠近末端）
- [ ] 传感器量程选择（预期力的2-3倍）
- [ ] 关节柔性是否需要建模
- [ ] 编码器分辨率满足精度要求
- [ ] 实时控制器延迟 < 1ms

#### 控制架构
- [ ] 控制频率 ≥ 1kHz（力控制）
- [ ] 多层控制结构（关节力矩、笛卡尔阻抗、任务规划）
- [ ] 故障检测与安全反应机制
- [ ] 模式切换逻辑完备
- [ ] 零空间利用策略（冗余机器人）

#### 参数设计
- [ ] 阻抗参数满足被动性条件
- [ ] 考虑工作空间内的参数变化
- [ ] 力控制增益 < 位置控制增益/100
- [ ] 滤波器截止频率选择合理
- [ ] 积分项限幅设置

#### 安全性
- [ ] 碰撞检测阈值设定
- [ ] 紧急停止逻辑
- [ ] 力/力矩限制
- [ ] 速度/加速度限制
- [ ] 故障模式分析（传感器失效、通信中断）

#### 性能验证
- [ ] 自由空间轨迹跟踪精度
- [ ] 接触力控制精度
- [ ] 阻抗调节范围测试
- [ ] 稳定性边界测试
- [ ] 鲁棒性测试（参数摄动、外部扰动）

#### 实施要点
- [ ] 重力补偿模型准确
- [ ] 摩擦补偿模型
- [ ] 传感器标定完成
- [ ] 坐标系定义清晰
- [ ] 代码实时性保证

#### 维护与调试
- [ ] 数据记录功能
- [ ] 参数在线调节接口
- [ ] 性能指标监控
- [ ] 故障日志系统
- [ ] 定期标定计划

---

*本章完成。阻抗控制是现代机器人实现安全、柔顺交互的核心技术。掌握其原理与实现细节，对于开发下一代智能机器人系统至关重要。*