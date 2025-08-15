# 第7章：动力学建模与参数辨识

## 学习目标

本章深入探讨轮足机械臂机器人的动力学建模方法与参数辨识技术。动力学模型是实现高性能控制的基础，它描述了力/力矩与运动之间的关系。我们将对比Newton-Euler和Lagrange两种主流建模方法，讨论柔性关节与传动系统的影响，分析摩擦力的建模与补偿策略，并介绍如何通过实验数据准确辨识动力学参数。通过本章学习，你将掌握构建精确动力学模型的完整流程，为实现基于模型的控制奠定基础。

## 7.1 刚体动力学基础

### 7.1.1 动力学方程的一般形式

机器人动力学方程描述了关节力矩与运动状态之间的关系，其一般形式为：

$$\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{G}(\mathbf{q}) + \mathbf{F}(\dot{\mathbf{q}})$$

其中：
- $\mathbf{M}(\mathbf{q}) \in \mathbb{R}^{n \times n}$：质量矩阵（惯性矩阵），对称正定
- $\mathbf{C}(\mathbf{q}, \dot{\mathbf{q}}) \in \mathbb{R}^{n \times n}$：科里奥利力和离心力矩阵
- $\mathbf{G}(\mathbf{q}) \in \mathbb{R}^{n}$：重力矩向量
- $\mathbf{F}(\dot{\mathbf{q}}) \in \mathbb{R}^{n}$：摩擦力矩向量
- $\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}} \in \mathbb{R}^{n}$：关节位置、速度、加速度

这个方程揭示了机器人动力学的几个关键特性：
1. **非线性耦合**：质量矩阵随配置变化，导致不同关节间的动力学耦合
2. **配置依赖**：重力项依赖于机器人姿态
3. **速度依赖**：科里奥利力与离心力产生速度相关的非线性项

### 7.1.2 Newton-Euler递推方法

Newton-Euler方法基于牛顿第二定律和欧拉方程，通过递推计算每个连杆的力和力矩。该方法计算效率高，复杂度为O(n)，适合实时控制。

**前向递推（运动学传播）**：
从基座向末端传递速度和加速度：

```
对于 i = 1 到 n：
    ω_{i} = R_{i}^{i-1} ω_{i-1} + ż_i θ̇_i
    α_{i} = R_{i}^{i-1} α_{i-1} + ż_i θ̈_i + ω_{i} × ż_i θ̇_i
    a_{ci} = α_{i} × r_{ci} + ω_{i} × (ω_{i} × r_{ci}) + a_{i}
```

**后向递推（动力学计算）**：
从末端向基座传递力和力矩：

```
对于 i = n 到 1：
    F_i = m_i a_{ci}
    N_i = I_{ci} α_i + ω_i × (I_{ci} ω_i)
    f_i = R_{i+1}^i f_{i+1} + F_i
    n_i = R_{i+1}^i n_{i+1} + N_i + r_{ci} × F_i + r_{i+1} × (R_{i+1}^i f_{i+1})
    τ_i = n_i^T z_i + b_i θ̇_i
```

其中：
- $ω_i, α_i$：连杆i的角速度和角加速度
- $a_{ci}$：连杆i质心的线加速度
- $F_i, N_i$：作用在连杆i质心的力和力矩
- $f_i, n_i$：关节i处的反作用力和力矩
- $I_{ci}$：连杆i相对质心的惯性张量

### 7.1.3 Lagrange方法

Lagrange方法基于能量原理，通过系统的动能和势能构建动力学方程。虽然符号计算复杂度较高，但物理意义清晰，便于理解系统的能量流动。

**拉格朗日函数**：
$$\mathcal{L}(\mathbf{q}, \dot{\mathbf{q}}) = T(\mathbf{q}, \dot{\mathbf{q}}) - V(\mathbf{q})$$

其中动能为：
$$T = \frac{1}{2}\sum_{i=1}^{n} \left( m_i \mathbf{v}_{ci}^T \mathbf{v}_{ci} + \boldsymbol{\omega}_i^T \mathbf{I}_{ci} \boldsymbol{\omega}_i \right)$$

势能为：
$$V = \sum_{i=1}^{n} m_i g^T \mathbf{p}_{ci}$$

**欧拉-拉格朗日方程**：
$$\tau_i = \frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \dot{q}_i}\right) - \frac{\partial \mathcal{L}}{\partial q_i}$$

展开后可得到标准形式的动力学方程。质量矩阵元素为：
$$M_{ij} = \sum_{k=\max(i,j)}^{n} \text{Tr}\left(\frac{\partial \mathbf{T}_k}{\partial q_i} \mathbf{I}_k \frac{\partial \mathbf{T}_k^T}{\partial q_j}\right)$$

### 7.1.4 两种方法的工程权衡

| 特性 | Newton-Euler | Lagrange |
|------|--------------|----------|
| 计算复杂度 | O(n) | O(n³)符号，O(n²)数值 |
| 物理直观性 | 力和力矩 | 能量 |
| 实现难度 | 中等，需要递推框架 | 简单，直接计算 |
| 适用场景 | 实时控制 | 离线分析、参数辨识 |
| 数值稳定性 | 优秀 | 良好 |
| 并行化潜力 | 有限（递推结构） | 高（独立计算项） |

在实际工程中，常采用混合策略：
- **控制器设计**：使用Lagrange方法推导解析表达式
- **实时计算**：使用Newton-Euler递推或预计算的Lagrange模型
- **参数辨识**：使用Lagrange方法构建线性回归模型

## 7.2 柔性关节与传动系统建模

### 7.2.1 柔性关节动力学

实际机器人的关节并非完全刚性，谐波减速器、同步带等传动元件引入了柔性。柔性关节模型将电机侧和连杆侧解耦，通过弹性元件连接：

$$\begin{aligned}
\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{G}(\mathbf{q}) &= \mathbf{K}(\boldsymbol{\theta} - \mathbf{q}) + \mathbf{D}(\dot{\boldsymbol{\theta}} - \dot{\mathbf{q}}) \\
\mathbf{B}\ddot{\boldsymbol{\theta}} + \mathbf{K}(\boldsymbol{\theta} - \mathbf{q}) + \mathbf{D}(\dot{\boldsymbol{\theta}} - \dot{\mathbf{q}}) &= \boldsymbol{\tau}_m
\end{aligned}$$

其中：
- $\boldsymbol{\theta}$：电机侧角度
- $\mathbf{q}$：连杆侧角度
- $\mathbf{B}$：电机转子惯量矩阵
- $\mathbf{K}$：关节刚度矩阵
- $\mathbf{D}$：关节阻尼矩阵
- $\boldsymbol{\tau}_m$：电机力矩

这个模型揭示了柔性关节的几个重要特性：
1. **共振频率**：$\omega_r = \sqrt{K/J_{eff}}$，其中$J_{eff}$是有效惯量
2. **反共振频率**：出现在电机到连杆的传递函数中
3. **相位滞后**：高频时连杆运动滞后于电机

### 7.2.2 传动系统建模

**谐波减速器模型**：
谐波减速器的非线性特性包括：
- **柔性**：扭转刚度约为$10^4 - 10^5$ Nm/rad
- **迟滞**：约占峰值扭矩的5-15%
- **动态摩擦**：随速度和负载变化

简化的迟滞模型（Bouc-Wen模型）：
$$\tau = K\theta + z$$
$$\dot{z} = A\dot{\theta} - \beta|\dot{\theta}||z|^{n-1}z - \gamma\dot{\theta}|z|^n$$

**同步带/链条传动**：
对于远距离传动，需考虑：
- **带的弹性**：等效刚度$K_{belt} = EA/L$
- **预紧力影响**：改变系统固有频率
- **多边形效应**：链条传动的速度波动

### 7.2.3 减速比与惯量匹配

传动系统的减速比n对系统动态特性有重要影响：

**反射惯量**：
$$J_{reflected} = J_{load}/n^2$$

**最优减速比**（最大加速度）：
$$n_{opt} = \sqrt{J_{load}/J_{motor}}$$

**带宽考虑**：
实际选择需权衡：
- 大减速比：增大力矩，但降低带宽
- 小减速比：提高响应速度，但需要大电机

工程经验法则：
```
轻载高速应用：n = 5-20
重载精密应用：n = 50-160
协作机器人：n = 80-120（平衡安全性与性能）
```

## 7.3 摩擦力建模与补偿

### 7.3.1 摩擦力模型

摩擦力是机器人控制中的主要非线性因素，准确建模对于实现高精度控制至关重要。

**静态摩擦模型**：

1. **库仑+粘性摩擦**：
$$F_f = F_c \cdot \text{sgn}(\dot{q}) + B_v \dot{q}$$

2. **Stribeck效应**：
$$F_f = \left[F_c + (F_s - F_c)e^{-|\dot{q}/\dot{q}_s|^{\delta}}\right]\text{sgn}(\dot{q}) + B_v \dot{q}$$

其中：
- $F_c$：库仑摩擦力
- $F_s$：静摩擦力（通常$F_s > F_c$）
- $B_v$：粘性摩擦系数
- $\dot{q}_s$：Stribeck速度
- $\delta$：形状参数（通常取2）

**动态摩擦模型（LuGre模型）**：

LuGre模型通过引入内部状态变量z描述接触面的弹性变形：

$$\begin{aligned}
\dot{z} &= \dot{q} - \frac{\sigma_0 |\dot{q}|}{g(\dot{q})} z \\
F_f &= \sigma_0 z + \sigma_1 \dot{z} + \sigma_2 \dot{q}
\end{aligned}$$

其中：
$$g(\dot{q}) = F_c + (F_s - F_c)e^{-|\dot{q}/\dot{q}_s|^2}$$

参数含义：
- $\sigma_0$：刚毛刚度
- $\sigma_1$：微观阻尼
- $\sigma_2$：粘性摩擦

### 7.3.2 摩擦力补偿策略

**基于模型的补偿**：
```
1. 离线辨识摩擦参数
2. 实时计算摩擦力估计值
3. 前馈补偿：τ_cmd = τ_desired + F_f_estimated
```

**自适应补偿**：
使用在线参数估计更新摩擦模型：
$$\hat{F}_f = \hat{\boldsymbol{\phi}}^T(\dot{q})\hat{\boldsymbol{\theta}}$$

参数更新律：
$$\dot{\hat{\boldsymbol{\theta}}} = -\boldsymbol{\Gamma}\boldsymbol{\phi}(\dot{q})s$$

其中s是滑模变量或跟踪误差。

**基于观测器的补偿**：
设计扰动观测器估计包括摩擦在内的总扰动：
$$\begin{aligned}
\hat{d} &= z + p(\dot{q}) \\
\dot{z} &= -L(q)[z + p(\dot{q})] + L(q)[\tau - M(q)\ddot{q}]
\end{aligned}$$

### 7.3.3 低速运动中的摩擦补偿

低速时摩擦力主导，需要特殊处理：

**抖动（Dither）信号**：
添加高频小幅振动打破静摩擦：
$$\tau_{dither} = A\sin(2\pi f_d t)$$

典型参数：
- 幅值A：静摩擦力的10-20%
- 频率$f_d$：100-500 Hz（高于控制带宽）

**脉冲控制**：
在过零点附近使用脉冲克服静摩擦：
```
if |velocity| < threshold and |position_error| > deadband:
    apply_pulse(sign(position_error) * pulse_amplitude)
```

## 7.4 参数辨识方法

### 7.4.1 动力学参数的线性化

虽然动力学方程关于运动状态是非线性的，但关于动力学参数（质量、质心、惯量）是线性的。这使得我们可以将动力学方程重写为：

$$\boldsymbol{\tau} = \mathbf{Y}(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}) \boldsymbol{\pi}$$

其中：
- $\mathbf{Y}$：回归矩阵（regressor matrix）
- $\boldsymbol{\pi}$：最小参数集（基参数）

**基参数的物理意义**：
每个连杆有10个标准惯性参数：
- 质量：$m_i$
- 质心位置：$mx_i, my_i, mz_i$
- 惯量张量：$I_{xx}, I_{xy}, I_{xz}, I_{yy}, I_{yz}, I_{zz}$

但并非所有参数都可独立辨识。基参数是可辨识的最小参数集，通常通过符号化简得到。

### 7.4.2 激励轨迹设计

辨识精度依赖于激励轨迹的设计。最优轨迹应最大化信息矩阵的某个指标。

**有限傅里叶级数轨迹**：
$$q_i(t) = q_{i0} + \sum_{k=1}^{N} \frac{a_{ik}}{k\omega} \sin(k\omega t) - \frac{b_{ik}}{k\omega} \cos(k\omega t)$$

优点：
- 周期性，便于多次采样平均
- 带宽可控
- 解析计算速度和加速度

**优化准则**：

1. **条件数最小化**：
$$\min \text{cond}(\mathbf{W}) = \min \frac{\sigma_{max}(\mathbf{W})}{\sigma_{min}(\mathbf{W})}$$

2. **D-优化（行列式最大化）**：
$$\max \det(\mathbf{W}^T\mathbf{W})$$

其中$\mathbf{W} = \int_0^T \mathbf{Y}^T\mathbf{Y} dt$是信息矩阵。

**约束条件**：
```
关节限位：q_min ≤ q(t) ≤ q_max
速度限制：|q̇(t)| ≤ q̇_max
加速度限制：|q̈(t)| ≤ q̈_max
力矩限制：|τ(t)| ≤ τ_max
```

### 7.4.3 参数估计算法

**最小二乘法（LS）**：
基本的参数估计：
$$\hat{\boldsymbol{\pi}} = (\mathbf{Y}^T\mathbf{Y})^{-1}\mathbf{Y}^T\boldsymbol{\tau}$$

**加权最小二乘（WLS）**：
考虑测量噪声的异方差性：
$$\hat{\boldsymbol{\pi}} = (\mathbf{Y}^T\mathbf{W}\mathbf{Y})^{-1}\mathbf{Y}^T\mathbf{W}\boldsymbol{\tau}$$

权重矩阵W通常选择为噪声协方差的逆。

**最大似然估计（MLE）**：
假设噪声模型：
$$\boldsymbol{\tau} = \mathbf{Y}\boldsymbol{\pi} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{\Sigma})$$

对数似然函数：
$$\mathcal{L} = -\frac{1}{2}(\boldsymbol{\tau} - \mathbf{Y}\boldsymbol{\pi})^T\boldsymbol{\Sigma}^{-1}(\boldsymbol{\tau} - \mathbf{Y}\boldsymbol{\pi})$$

**贝叶斯估计**：
引入参数的先验分布：
$$p(\boldsymbol{\pi}) = \mathcal{N}(\boldsymbol{\pi}_0, \boldsymbol{\Sigma}_0)$$

后验分布：
$$p(\boldsymbol{\pi}|\boldsymbol{\tau}) \propto p(\boldsymbol{\tau}|\boldsymbol{\pi})p(\boldsymbol{\pi})$$

MAP估计：
$$\hat{\boldsymbol{\pi}}_{MAP} = \arg\max_{\boldsymbol{\pi}} [p(\boldsymbol{\tau}|\boldsymbol{\pi})p(\boldsymbol{\pi})]$$

### 7.4.4 在线辨识与自适应

**递推最小二乘（RLS）**：
```
初始化：P(0) = αI, θ̂(0) = θ_0
对于每个新样本(y_k, φ_k)：
    e_k = y_k - φ_k^T θ̂_{k-1}
    K_k = P_{k-1}φ_k / (λ + φ_k^T P_{k-1} φ_k)
    θ̂_k = θ̂_{k-1} + K_k e_k
    P_k = (I - K_k φ_k^T)P_{k-1}/λ
```

其中λ是遗忘因子（0.95-0.99）。

**扩展卡尔曼滤波（EKF）辨识**：
将参数作为状态变量：
$$\begin{aligned}
\mathbf{x} &= [\mathbf{q}^T, \dot{\mathbf{q}}^T, \boldsymbol{\pi}^T]^T \\
\dot{\mathbf{x}} &= f(\mathbf{x}, \mathbf{u}) + \mathbf{w}
\end{aligned}$$

测量方程：
$$\mathbf{z} = h(\mathbf{x}) + \mathbf{v}$$

### 7.4.5 物理一致性约束

辨识的参数应满足物理约束：

**质量正定性**：
$$m_i > 0$$

**惯量矩阵正定性**：
$$\mathbf{I}_i = \begin{bmatrix}
I_{xx} & I_{xy} & I_{xz} \\
I_{xy} & I_{yy} & I_{yz} \\
I_{xz} & I_{yz} & I_{zz}
\end{bmatrix} \succ 0$$

**三角不等式**：
$$\begin{aligned}
I_{xx} + I_{yy} &\geq I_{zz} \\
I_{yy} + I_{zz} &\geq I_{xx} \\
I_{zz} + I_{xx} &\geq I_{yy}
\end{aligned}$$

**质心位置约束**：
$$\|\mathbf{r}_{ci}\| \leq r_{max}$$

这些约束可通过约束优化方法（如SDP）强制满足。

## 案例研究：KUKA iiwa的动力学补偿

KUKA iiwa是一款7自由度协作机器人，以其精确的动力学补偿和力矩控制能力著称。

### 系统架构

iiwa采用分层控制架构：
1. **位置控制器**（1 kHz）：笛卡尔/关节空间轨迹跟踪
2. **动力学补偿**（3 kHz）：重力、科里奥利力实时计算
3. **力矩控制器**（10 kHz）：电流环控制

### 动力学模型特点

**谐波减速器建模**：
- 每个关节配备高精度扭矩传感器
- 实测刚度：$10^4 - 10^5$ Nm/rad
- 考虑温度对刚度的影响（约20%变化）

**摩擦补偿**：
- 使用分段线性模型近似Stribeck曲线
- 温度补偿：$F_f(T) = F_f(T_0)[1 + \alpha(T-T_0)]$
- 负载自适应：根据负载调整摩擦参数

### 参数辨识流程

1. **预热程序**：运行标准轨迹30分钟达到热平衡
2. **激励轨迹**：5阶傅里叶级数，基频0.05 Hz
3. **数据采集**：10 kHz采样，低通滤波至100 Hz
4. **迭代辨识**：
   - 第一轮：辨识惯性参数
   - 第二轮：固定惯性参数，辨识摩擦
   - 第三轮：联合优化

### 性能指标

通过精确的动力学补偿，iiwa实现了：
- 重复定位精度：±0.1 mm
- 力矩估计误差：< 2% 额定力矩
- 外力检测阈值：< 2 N（末端）
- 重力补偿精度：< 0.5% 负载重量

### 工程启示

1. **传感器融合**：结合电机编码器和关节扭矩传感器提高估计精度
2. **温度补偿**：工业应用必须考虑温度变化
3. **在线更新**：通过持续学习适应磨损和老化
4. **安全裕度**：参数不确定性纳入安全策略设计

## 高级话题：接触动力学与LCP

### 接触力计算

多点接触时，接触力需满足：
1. **非穿透约束**：$\phi_i(\mathbf{q}) \geq 0$
2. **非负法向力**：$f_{n,i} \geq 0$
3. **互补条件**：$\phi_i f_{n,i} = 0$
4. **摩擦锥约束**：$\|\mathbf{f}_{t,i}\| \leq \mu f_{n,i}$

这构成一个线性互补问题（LCP）：
$$\begin{aligned}
\mathbf{w} &= \mathbf{M}\mathbf{z} + \mathbf{q} \\
\mathbf{w} &\geq 0, \quad \mathbf{z} \geq 0 \\
\mathbf{w}^T\mathbf{z} &= 0
\end{aligned}$$

### 求解方法

**Lemke算法**：
主元算法，保证有限步收敛，但可能遇到退化。

**投影Gauss-Seidel（PGS）**：
```
for iteration = 1 to max_iter:
    for each contact i:
        z_i = max(0, z_i - ω(Mz + q)_i/M_{ii})
```

收敛速度依赖于松弛因子ω的选择。

**基于优化的方法**：
将LCP转换为二次规划（QP）：
$$\min_{\mathbf{z}} \frac{1}{2}\mathbf{z}^T\mathbf{M}\mathbf{z} + \mathbf{q}^T\mathbf{z}, \quad \text{s.t. } \mathbf{z} \geq 0$$

### 工程考虑

1. **刚度选择**：过大导致数值不稳定，过小导致穿透
2. **时间步长**：显式积分需要小步长（< 1 ms）
3. **正则化**：添加阻尼项改善条件数
4. **接触检测**：使用空间哈希或BVH加速

## 本章小结

本章系统介绍了轮足机械臂机器人的动力学建模与参数辨识：

**关键概念**：
1. 动力学方程的一般形式：$\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{G}(\mathbf{q}) + \mathbf{F}(\dot{\mathbf{q}})$
2. Newton-Euler递推：O(n)复杂度，适合实时控制
3. Lagrange方法：基于能量，便于符号推导
4. 柔性关节模型：考虑传动系统柔性的双惯量模型
5. 摩擦模型：从简单的库仑+粘性到复杂的LuGre模型
6. 参数线性化：$\boldsymbol{\tau} = \mathbf{Y}(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}}) \boldsymbol{\pi}$
7. 激励轨迹优化：最大化信息矩阵的可观测性

**实践要点**：
- 根据控制需求选择合适的模型复杂度
- 摩擦补偿对低速精度至关重要
- 参数辨识需要精心设计的激励轨迹
- 物理一致性约束确保辨识结果合理
- 温度和负载变化需要在线适应

掌握这些概念和方法，你将能够为机器人构建精确的动力学模型，这是实现高性能控制的基础。下一章我们将探讨如何利用这些模型进行轨迹规划与优化。

## 练习题

### 基础题

**练习7.1** 推导2-DOF平面机械臂的动力学方程
考虑一个2自由度平面机械臂，连杆长度为$l_1, l_2$，质量为$m_1, m_2$，质心位于连杆中点。使用Lagrange方法推导其动力学方程，并识别质量矩阵、科里奥利矩阵和重力向量。

*提示：先写出每个连杆质心的位置，计算动能和势能。*

<details>
<summary>参考答案</summary>

质心位置：
- 连杆1：$x_{c1} = \frac{l_1}{2}\cos q_1, y_{c1} = \frac{l_1}{2}\sin q_1$
- 连杆2：$x_{c2} = l_1\cos q_1 + \frac{l_2}{2}\cos(q_1+q_2), y_{c2} = l_1\sin q_1 + \frac{l_2}{2}\sin(q_1+q_2)$

动能：
$$T = \frac{1}{2}[m_1(\frac{l_1}{2})^2 + I_1]\dot{q}_1^2 + \frac{1}{2}[m_2l_1^2\dot{q}_1^2 + m_2(\frac{l_2}{2})^2(\dot{q}_1+\dot{q}_2)^2 + I_2(\dot{q}_1+\dot{q}_2)^2 + 2m_2l_1\frac{l_2}{2}\cos q_2\dot{q}_1(\dot{q}_1+\dot{q}_2)]$$

质量矩阵：
$$\mathbf{M} = \begin{bmatrix}
m_1(\frac{l_1}{2})^2 + m_2l_1^2 + m_2(\frac{l_2}{2})^2 + I_1 + I_2 + m_2l_1l_2\cos q_2 & m_2(\frac{l_2}{2})^2 + I_2 + \frac{1}{2}m_2l_1l_2\cos q_2 \\
m_2(\frac{l_2}{2})^2 + I_2 + \frac{1}{2}m_2l_1l_2\cos q_2 & m_2(\frac{l_2}{2})^2 + I_2
\end{bmatrix}$$

科里奥利矩阵：
$$\mathbf{C} = \begin{bmatrix}
-m_2l_1\frac{l_2}{2}\sin q_2 \dot{q}_2 & -m_2l_1\frac{l_2}{2}\sin q_2(\dot{q}_1+\dot{q}_2) \\
m_2l_1\frac{l_2}{2}\sin q_2 \dot{q}_1 & 0
\end{bmatrix}$$

重力向量：
$$\mathbf{G} = \begin{bmatrix}
(m_1\frac{l_1}{2} + m_2l_1)g\cos q_1 + m_2\frac{l_2}{2}g\cos(q_1+q_2) \\
m_2\frac{l_2}{2}g\cos(q_1+q_2)
\end{bmatrix}$$
</details>

**练习7.2** Newton-Euler递推实现
编写伪代码实现3-DOF机械臂的Newton-Euler递推算法。假设已知每个连杆的质量$m_i$、质心位置$\mathbf{r}_{ci}$、惯性张量$\mathbf{I}_i$，以及当前的关节位置、速度和加速度。

*提示：分别实现前向运动学递推和后向动力学递推。*

<details>
<summary>参考答案</summary>

```
// 前向递推
ω[0] = 0, α[0] = 0, a[0] = -g
for i = 1 to 3:
    R[i] = Rz(q[i]) @ R[i-1]  // 旋转矩阵
    ω[i] = R[i]^T @ ω[i-1] + [0, 0, q̇[i]]
    α[i] = R[i]^T @ α[i-1] + [0, 0, q̈[i]] + cross(ω[i], [0, 0, q̇[i]])
    a[i] = R[i]^T @ (a[i-1] + cross(α[i-1], p[i]) + cross(ω[i-1], cross(ω[i-1], p[i])))
    ac[i] = a[i] + cross(α[i], rc[i]) + cross(ω[i], cross(ω[i], rc[i]))

// 后向递推
f[4] = 0, n[4] = 0  // 末端无外力
for i = 3 to 1:
    F[i] = m[i] * ac[i]
    N[i] = I[i] @ α[i] + cross(ω[i], I[i] @ ω[i])
    f[i] = R[i+1] @ f[i+1] + F[i]
    n[i] = R[i+1] @ n[i+1] + cross(rc[i], F[i]) + cross(p[i+1], R[i+1] @ f[i+1]) + N[i]
    τ[i] = n[i] · [0, 0, 1]  // 提取z分量
```
</details>

**练习7.3** 摩擦力参数辨识
给定一组低速运动数据（速度范围：-0.1到0.1 rad/s），设计一个方法辨识Stribeck摩擦模型的参数。数据格式为$(
\dot{q}_i, \tau_{friction,i})$。

*提示：使用非线性最小二乘拟合。*

<details>
<summary>参考答案</summary>

Stribeck模型：
$$F_f = [F_c + (F_s - F_c)e^{-|\dot{q}/\dot{q}_s|^2}]\text{sgn}(\dot{q}) + B_v\dot{q}$$

辨识步骤：
1. 分离正负速度数据
2. 初始参数估计：
   - $F_s$：$\dot{q} \approx 0$时的摩擦力
   - $F_c$：高速时的摩擦力渐近值
   - $B_v$：高速区域的斜率
   - $\dot{q}_s$：Stribeck下降区的特征速度

3. 非线性优化：
```python
def objective(params, q_dot, tau_measured):
    Fc, Fs, Bv, qs = params
    tau_pred = (Fc + (Fs-Fc)*exp(-(q_dot/qs)**2))*sign(q_dot) + Bv*q_dot
    return sum((tau_measured - tau_pred)**2)

params_opt = minimize(objective, params_init, args=(q_dot_data, tau_data))
```

4. 验证对称性：正负方向参数应相近
</details>

**练习7.4** 基参数计算
对于一个3-DOF机械臂，原始有30个惯性参数（每个连杆10个）。说明如何通过符号运算确定基参数集，并给出可能的基参数数量。

*提示：考虑哪些参数组合总是一起出现在动力学方程中。*

<details>
<summary>参考答案</summary>

基参数确定方法：
1. 写出符号形式的动力学方程
2. 识别线性相关的参数组合
3. 常见的参数组合：
   - 第一个连杆的$I_{zz,1} + m_2l_1^2 + m_3l_1^2$总是一起出现
   - 末端连杆绕其关节轴的惯量独立
   - 零重力方向的质心坐标不可辨识

对于3-DOF平面机械臂：
- 原始参数：30个
- 不可辨识：沿重力方向的质心坐标（3个）
- 组合参数：约6-8个
- 基参数数量：通常为12-15个

具体数量取决于：
- 机械臂构型（串联/并联）
- 关节类型（旋转/移动）
- 重力方向
</details>

### 挑战题

**练习7.5** 柔性关节控制器设计
考虑单自由度柔性关节：
$$\begin{aligned}
J_l\ddot{q} + mgl\sin(q) &= k(\theta - q) \\
J_m\ddot{\theta} &= \tau - k(\theta - q)
\end{aligned}$$

其中$J_l=1$, $J_m=0.1$, $k=100$, $mgl=10$。设计一个控制器实现位置跟踪，分析系统的稳定性和振动抑制。

*提示：考虑基于奇异摄动理论的双时间尺度控制。*

<details>
<summary>参考答案</summary>

系统分析：
1. 固有频率：
   - 刚体模式：$\omega_1 \approx \sqrt{mgl/J_l} \approx 3.16$ rad/s
   - 弹性模式：$\omega_2 = \sqrt{k(1/J_m + 1/J_l)} \approx 33.2$ rad/s

2. 双时间尺度控制：
   - 慢子系统（刚体）：忽略弹性，设计PD控制器
   - 快子系统（振动）：主动阻尼控制

控制律：
$$\tau = k(\theta_d - \theta) + b(\dot{\theta}_d - \dot{\theta}) + k(q - \theta) + \tau_{ff}$$

其中：
- 前馈：$\tau_{ff} = mgl\sin(q_d)$
- 位置环：$\theta_d = q_d + (mgl\sin(q_d))/k$
- 阻尼：$b = 2\zeta\sqrt{kJ_m}$，$\zeta \approx 0.7$

稳定性分析：
- 使用Lyapunov函数证明渐近稳定
- 通过根轨迹分析选择增益
</details>

**练习7.6** 在线参数自适应
设计一个自适应控制器，同时跟踪期望轨迹并在线估计未知的负载质量。考虑2-DOF机械臂末端抓取未知质量$m_L$的负载。

*提示：使用Slotine-Li自适应控制方法。*

<details>
<summary>参考答案</summary>

自适应控制律：
$$\boldsymbol{\tau} = \hat{\mathbf{M}}\ddot{\mathbf{q}}_r + \hat{\mathbf{C}}\dot{\mathbf{q}}_r + \hat{\mathbf{G}} + \mathbf{K}_D\mathbf{s}$$

其中：
- 滑模变量：$\mathbf{s} = \dot{\tilde{\mathbf{q}}} + \boldsymbol{\Lambda}\tilde{\mathbf{q}}$
- 参考速度：$\dot{\mathbf{q}}_r = \dot{\mathbf{q}}_d - \boldsymbol{\Lambda}\tilde{\mathbf{q}}$

参数更新律：
$$\dot{\hat{m}}_L = -\gamma \mathbf{Y}_L^T(\mathbf{q}, \dot{\mathbf{q}}_r, \ddot{\mathbf{q}}_r)\mathbf{s}$$

其中$\mathbf{Y}_L$是负载质量的回归向量。

Lyapunov函数：
$$V = \frac{1}{2}\mathbf{s}^T\mathbf{M}\mathbf{s} + \frac{1}{2\gamma}\tilde{m}_L^2$$

证明$\dot{V} \leq -\mathbf{s}^T\mathbf{K}_D\mathbf{s} \leq 0$，系统全局渐近稳定。

实现细节：
- 参数投影保证$\hat{m}_L \geq 0$
- 死区修正避免噪声驱动
- 持续激励条件保证参数收敛
</details>

**练习7.7** 多接触场景的LCP求解
机器人手抓握一个立方体，有4个接触点，每个接触点有3维接触力（1个法向，2个切向）。建立LCP问题并说明求解策略。摩擦系数$\mu=0.3$。

*提示：使用摩擦锥的线性化近似。*

<details>
<summary>参考答案</summary>

问题建立：
1. 变量：12个接触力分量
2. 动力学约束：
   $$\mathbf{M}\ddot{\mathbf{q}} = \mathbf{J}_c^T\mathbf{f}_c + \boldsymbol{\tau}$$

3. 接触约束：
   - 非穿透：$\phi_i \geq 0$
   - 非负法向力：$f_{n,i} \geq 0$
   - 互补性：$\phi_i f_{n,i} = 0$

4. 摩擦锥线性化（4面锥）：
   $$\mathbf{D}_i\mathbf{f}_{t,i} \leq \mu f_{n,i}\mathbf{e}$$
   
   其中$\mathbf{D}_i$定义4个摩擦面。

LCP形式：
$$\begin{bmatrix}
\mathbf{A} & -\mathbf{J}_c^T \\
\mathbf{J}_c & \mathbf{0}
\end{bmatrix}
\begin{bmatrix}
\ddot{\mathbf{q}} \\
\boldsymbol{\lambda}
\end{bmatrix}
+
\begin{bmatrix}
\mathbf{b} \\
\mathbf{c}
\end{bmatrix}
= \mathbf{0}$$

求解策略：
1. **PGS迭代**：适合实时，但收敛慢
2. **PATH求解器**：鲁棒但需要许可证
3. **内点法**：转化为QP问题

实现建议：
- 预处理改善条件数
- 温启动加速收敛
- 正则化处理奇异配置
</details>

**练习7.8** 接触参数的在线估计
设计一个算法在线估计接触刚度和阻尼系数。假设机器人末端与环境接触，接触模型为：
$$F_c = k_e(x - x_e) + b_e\dot{x}$$

*提示：使用递推最小二乘或卡尔曼滤波。*

<details>
<summary>参考答案</summary>

扩展卡尔曼滤波方法：

状态向量：
$$\mathbf{x} = [x, \dot{x}, k_e, b_e, x_e]^T$$

系统模型：
$$\begin{aligned}
\dot{x} &= v \\
\dot{v} &= (F_{cmd} - k_e(x-x_e) - b_e v)/m \\
\dot{k}_e &= 0 + w_{k} \\
\dot{b}_e &= 0 + w_{b} \\
\dot{x}_e &= 0 + w_{x}
\end{aligned}$$

测量模型：
- 位置：$z_x = x + v_x$
- 力：$z_F = k_e(x-x_e) + b_e\dot{x} + v_F$

EKF更新：
1. 预测：
   $$\hat{\mathbf{x}}_{k|k-1} = f(\hat{\mathbf{x}}_{k-1}, u_{k-1})$$
   $$\mathbf{P}_{k|k-1} = \mathbf{F}_k\mathbf{P}_{k-1}\mathbf{F}_k^T + \mathbf{Q}$$

2. 更新：
   $$\mathbf{K}_k = \mathbf{P}_{k|k-1}\mathbf{H}_k^T(\mathbf{H}_k\mathbf{P}_{k|k-1}\mathbf{H}_k^T + \mathbf{R})^{-1}$$
   $$\hat{\mathbf{x}}_{k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k(\mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1}))$$

关键参数：
- 过程噪声$\mathbf{Q}$：决定参数变化速度
- 测量噪声$\mathbf{R}$：基于传感器精度
- 初始协方差：反映参数不确定性

验证方法：
- 施加阶跃力，观察参数收敛
- 正弦激励，验证频率响应
</details>

## 常见陷阱与错误

1. **数值积分不稳定**
   - 错误：使用显式Euler积分刚性系统
   - 正确：使用隐式方法或可变步长求解器

2. **忽略耦合效应**
   - 错误：独立设计各关节控制器
   - 正确：考虑惯量矩阵的非对角项

3. **摩擦补偿过度**
   - 错误：完全消除摩擦
   - 正确：保留适当阻尼避免振荡

4. **参数辨识局部最优**
   - 错误：使用单一初始值
   - 正确：多次随机初始化或全局优化

5. **忽略模型不确定性**
   - 错误：假设模型完美
   - 正确：设计鲁棒控制器处理参数变化

6. **激励信号设计不当**
   - 错误：使用单频正弦信号
   - 正确：宽频激励确保可辨识性

## 最佳实践检查清单

### 建模阶段
- [ ] 选择合适的坐标系和参数化方法
- [ ] 验证质量矩阵的对称性和正定性
- [ ] 检查能量守恒（无摩擦无控制时）
- [ ] 考虑关节限位和饱和

### 参数辨识
- [ ] 设计覆盖工作空间的激励轨迹
- [ ] 多次实验取平均减少噪声
- [ ] 验证参数的物理一致性
- [ ] 交叉验证辨识结果

### 实时实现
- [ ] 选择合适的采样频率（> 10倍带宽）
- [ ] 优化计算（查表、并行化）
- [ ] 实现故障检测和安全停机
- [ ] 记录数据用于离线分析

### 性能验证
- [ ] 测试不同负载和速度
- [ ] 评估跟踪精度和稳定裕度
- [ ] 长时间运行测试
- [ ] 极限工况测试