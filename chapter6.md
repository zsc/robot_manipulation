# 第6章：正逆运动学与工作空间

运动学是机器人控制的基石，它建立了关节空间与任务空间之间的映射关系。对于轮足机械臂这样的复杂系统，运动学不仅决定了机器人能够到达的位置，更影响着运动的灵活性、奇异性规避以及多任务协调能力。本章将深入探讨正逆运动学的求解方法，重点关注工程实践中的数值稳定性、计算效率和冗余度利用，为后续的动力学控制和智能规划奠定基础。

## 6.1 正运动学基础

### 6.1.1 运动学链与坐标系定义

轮足机械臂系统的运动学链通常包含三个主要部分：移动底盘、腿部机构和机械臂。每个部分都有其独特的运动学特性：

- **移动底盘**：提供全局定位能力，其运动学模型需要考虑轮子的滑移和地形适应
- **腿部机构**：实现高度调节和姿态控制，通常采用并联或串并混联结构
- **机械臂**：执行精确操作任务，多为串联结构with 6-7个自由度

完整的运动学链可以表示为：

$$\mathbf{T}_{base}^{ee} = \mathbf{T}_{world}^{base} \cdot \mathbf{T}_{base}^{leg} \cdot \mathbf{T}_{leg}^{arm} \cdot \mathbf{T}_{arm}^{ee}$$

其中$\mathbf{T}$表示齐次变换矩阵，下标表示坐标系，上标表示变换目标。

在实际工程中，坐标系的选择至关重要。不当的坐标系定义会导致控制算法复杂化，甚至引入不必要的数值问题。轮足机器人的坐标系层次通常包括：

1. **世界坐标系**$\{W\}$：固定在环境中的绝对参考系，用于全局路径规划和定位
2. **机器人基座坐标系**$\{B\}$：随机器人移动，但不随腿部运动改变
3. **浮动基坐标系**$\{F\}$：位于机器人质心或腰部，用于动力学计算
4. **关节坐标系**$\{J_i\}$：每个关节的局部坐标系，用于描述关节运动
5. **末端执行器坐标系**$\{E\}$：工具坐标系，定义抓取或操作参考点

坐标系间的相对关系需要实时更新，特别是在不平整地形上运动时。传感器融合算法（如扩展卡尔曼滤波）用于估计$\mathbf{T}_{world}^{base}$，而编码器直接提供关节角度用于计算其他变换。

### 6.1.2 DH参数与齐次变换

尽管Product of Exponentials (PoE)方法在理论上更优雅，但DH参数因其标准化和直观性在工业界仍广泛使用。对于第$i$个关节，DH变换矩阵为：

$$\mathbf{A}_i = \begin{bmatrix}
\cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\
\sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\
0 & \sin\alpha_i & \cos\alpha_i & d_i \\
0 & 0 & 0 & 1
\end{bmatrix}$$

其中：
- $\theta_i$：关节角（旋转关节的变量）
- $d_i$：连杆偏移（移动关节的变量）
- $a_i$：连杆长度（固定参数）
- $\alpha_i$：连杆扭转（固定参数）

DH参数的选择并非唯一，存在标准DH（Standard DH）和修正DH（Modified DH）两种约定。标准DH将坐标系原点放在当前关节轴上，而修正DH则放在前一关节轴上。这种差异看似细微，但会影响雅可比矩阵的计算和奇异性分析。

**DH参数标定的实践考虑**：

实际机器人的DH参数与理论设计值存在偏差，原因包括：
- 制造公差和装配误差
- 关节轴线不完全平行或垂直
- 重力和负载引起的结构变形
- 温度变化导致的热膨胀

参数标定通常采用以下方法：
1. **几何标定**：使用激光跟踪仪或视觉系统测量末端位置
2. **运动学标定**：通过圆形或直线运动轨迹拟合参数
3. **自标定**：利用冗余传感器信息（如关节扭矩）推算参数

标定误差模型：
$$\Delta \mathbf{x} = \mathbf{J}_{error} \Delta \boldsymbol{\phi}$$

其中$\Delta \boldsymbol{\phi}$是参数误差向量，$\mathbf{J}_{error}$是误差雅可比矩阵。通过最小二乘法可以辨识参数偏差：
$$\Delta \boldsymbol{\phi} = (\mathbf{J}_{error}^T \mathbf{J}_{error})^{-1} \mathbf{J}_{error}^T \Delta \mathbf{x}$$

### 6.1.3 轮足平台的特殊考虑

轮足机械臂的正运动学需要处理几个特殊问题：

1. **浮动基座**：底盘位置不固定，需要通过里程计或SLAM实时更新
2. **接触约束**：腿部接地时引入额外的运动学约束
3. **变拓扑结构**：行走模式切换时运动学链发生变化

```
     机械臂末端
         |
    [机械臂关节链]
         |
     腿部平台
       /   \
   左腿链  右腿链
     |       |
   左轮     右轮
     \     /
      地面
```

**浮动基座的处理**：

传统固定基座机器人的运动学从基座开始，而轮足机器人的基座本身在运动。这需要引入额外的状态估计：

$$\mathbf{T}_{world}^{base}(t) = \begin{bmatrix}
\mathbf{R}(t) & \mathbf{p}(t) \\
\mathbf{0} & 1
\end{bmatrix}$$

其中$\mathbf{R}(t) \in SO(3)$是基座姿态，$\mathbf{p}(t) \in \mathbb{R}^3$是基座位置。这些量通过以下方式估计：

- **轮式里程计**：积分轮速得到位移，但存在累积误差
- **视觉里程计**：通过图像特征跟踪估计运动
- **IMU积分**：提供短期精确的姿态和加速度信息
- **融合算法**：如EKF、UKF或因子图优化

**接触约束的建模**：

当腿部与地面接触时，形成运动学闭链。设接触点位置为$\mathbf{c}_i$，约束方程为：

$$\mathbf{T}_{world}^{base} \cdot \mathbf{T}_{base}^{foot_i}(\mathbf{q}_{leg}) = \mathbf{T}_{world}^{contact_i}$$

这引入了$3m$个约束（$m$是接触点数），减少了系统的有效自由度。在运动规划时必须确保约束的满足：

$$\mathbf{J}_{contact} \dot{\mathbf{q}} = \mathbf{0}$$

**模式切换的连续性**：

轮足机器人在不同运动模式间切换（如从轮式到足式），运动学模型会发生突变。平滑过渡策略包括：

1. **过渡期重叠**：在模式切换前后保持双模式约束
2. **虚拟关节**：引入柔性虚拟关节吸收不连续性
3. **轨迹混合**：使用样条插值确保位置和速度连续

## 6.2 逆运动学求解策略

### 6.2.1 解析解方法

对于特定的机械臂构型（如球形腕、拟人臂），可以推导闭式解析解。以6自由度机械臂为例，常用的解析方法包括：

**几何法**：利用机械臂的几何特征分解问题
1. 位置解耦：先求解前3个关节确定腕部位置
2. 姿态解耦：后3个关节确定末端姿态

**代数法**：通过矩阵方程系统求解
1. 构建约束方程：$\mathbf{T}_0^6(\theta) = \mathbf{T}_{desired}$
2. 逐步消元求解各关节角

解析解的优势在于计算速度快、精度高，但仅适用于特定构型。大多数冗余机械臂没有闭式解。

**Pieper准则**：

对于6自由度串联机械臂，存在闭式解的充分条件是满足Pieper准则之一：
1. 三个相邻关节轴相交于一点（球形腕）
2. 三个相邻关节轴平行

大多数工业机器人采用球形腕设计正是为了保证解析解的存在性。对于满足Pieper准则的机械臂，求解步骤为：

**步骤1：腕心位置求解**
$$\mathbf{p}_{wrist} = \mathbf{p}_{desired} - d_6 \mathbf{R}_{desired} \mathbf{z}_0$$

其中$d_6$是腕心到末端的距离，$\mathbf{z}_0 = [0, 0, 1]^T$。

**步骤2：前三关节求解**

利用腕心位置的几何约束：
$$|\mathbf{p}_{wrist}|^2 = (a_2 \cos\theta_2 + a_3)^2 + (a_2 \sin\theta_2 + d_4)^2$$

可得$\theta_2$的解：
$$\cos\theta_2 = \frac{|\mathbf{p}_{wrist}|^2 - a_2^2 - a_3^2 - d_4^2}{2a_2\sqrt{a_3^2 + d_4^2}}$$

**步骤3：后三关节求解**

利用姿态约束：
$$\mathbf{R}_3^6(\theta_4, \theta_5, \theta_6) = (\mathbf{R}_0^3)^T \mathbf{R}_{desired}$$

通过欧拉角分解可得$\theta_4, \theta_5, \theta_6$。

### 6.2.2 数值迭代方法

对于一般构型和冗余机械臂，数值方法是主要选择：

**牛顿-拉夫逊法**：
$$\Delta\mathbf{q} = \mathbf{J}^{-1}(\mathbf{q}) \cdot \Delta\mathbf{x}$$

其中$\mathbf{J}$是雅可比矩阵，$\Delta\mathbf{x}$是笛卡尔空间误差，$\Delta\mathbf{q}$是关节空间增量。

**阻尼最小二乘法（DLS）**：
$$\Delta\mathbf{q} = \mathbf{J}^T(\mathbf{J}\mathbf{J}^T + \lambda^2\mathbf{I})^{-1} \cdot \Delta\mathbf{x}$$

阻尼因子$\lambda$提供数值稳定性，特别是在奇异点附近。自适应阻尼策略：

$$\lambda = \begin{cases}
0 & \text{if } \sigma_{min} > \epsilon \\
\lambda_0 \sqrt{1 - (\sigma_{min}/\epsilon)^2} & \text{otherwise}
\end{cases}$$

其中$\sigma_{min}$是雅可比矩阵的最小奇异值。

**改进的数值方法**：

1. **Levenberg-Marquardt方法**：
结合了高斯-牛顿法和梯度下降法的优点：
$$(\mathbf{J}^T\mathbf{J} + \mu\mathbf{I})\Delta\mathbf{q} = -\mathbf{J}^T\mathbf{e}$$

其中$\mu$根据收敛情况动态调整：
- 收敛快时减小$\mu$（更像高斯-牛顿）
- 收敛慢时增大$\mu$（更像梯度下降）

2. **BFGS拟牛顿法**：
避免直接计算雅可比矩阵，通过迭代更新Hessian近似：
$$\mathbf{H}_{k+1} = \mathbf{H}_k + \frac{\mathbf{y}\mathbf{y}^T}{\mathbf{y}^T\mathbf{s}} - \frac{\mathbf{H}_k\mathbf{s}\mathbf{s}^T\mathbf{H}_k}{\mathbf{s}^T\mathbf{H}_k\mathbf{s}}$$

3. **信赖域方法**：
限制每步的移动范围，提高鲁棒性：
$$\min_{\Delta\mathbf{q}} ||\mathbf{J}\Delta\mathbf{q} + \mathbf{e}||^2 \quad \text{s.t.} \quad ||\Delta\mathbf{q}|| \leq \delta$$

**收敛性分析**：

数值方法的收敛速度取决于多个因素：
- **初值选择**：接近解的初值可显著加快收敛
- **条件数**：$\kappa(\mathbf{J})$越大收敛越慢
- **步长选择**：过大导致振荡，过小收敛慢

典型的收敛准则组合：
$$||\mathbf{e}|| < \epsilon_{pos} \quad \text{且} \quad ||\Delta\mathbf{q}|| < \epsilon_{joint} \quad \text{且} \quad k < k_{max}$$

### 6.2.3 多解处理与最优解选择

逆运动学通常有多个解（如肘部上下、腕部翻转等）。选择策略包括：

1. **最小关节运动**：$\min ||\mathbf{q} - \mathbf{q}_{current}||$
2. **避免关节极限**：$\max \min_i \{(q_i - q_{i,min}), (q_{i,max} - q_i)\}$
3. **操作性优化**：$\max \sqrt{\det(\mathbf{J}\mathbf{J}^T)}$
4. **任务相关准则**：如最小化特定关节负载

**多解的几何解释**：

对于典型的6-DOF机械臂，最多可能有16个解：
- 肩部：左/右配置（2种）
- 肘部：上/下配置（2种）
- 腕部：翻转/非翻转（2种）
- 每种组合可能有多个腕部姿态解（最多2种）

总计：$2 \times 2 \times 2 \times 2 = 16$个可能解。

**解的连续性管理**：

在连续运动中，解的选择必须保持一致性，避免关节突变：

```
解分支跟踪算法：
1. 计算所有可能解 {q_1, q_2, ..., q_n}
2. 对每个解计算连续性度量：
   c_i = exp(-||q_i - q_prev||/σ)
3. 选择得分最高的解：
   q_selected = argmax(c_i · w_i)
   其中w_i是任务相关权重
```

**全局最优解搜索**：

对于关键任务，可以通过优化方法寻找全局最优：

$$\mathbf{q}^* = \arg\min_{\mathbf{q}} \left( w_1 f_{joint}(\mathbf{q}) + w_2 f_{limit}(\mathbf{q}) + w_3 f_{manip}(\mathbf{q}) + w_4 f_{energy}(\mathbf{q}) \right)$$

其中：
- $f_{joint}$：关节运动成本
- $f_{limit}$：关节极限惩罚
- $f_{manip}$：操作性度量
- $f_{energy}$：能耗估计

**解的可达性预测**：

在开始IK求解前，快速判断目标是否可达：
1. **距离检查**：$||\mathbf{p}_{target}|| \leq r_{max}$
2. **工作空间检查**：点在凸包内
3. **关节极限预估**：基于简化模型的快速检查

## 6.3 雅可比矩阵与微分运动学

### 6.3.1 雅可比矩阵的计算

雅可比矩阵建立了关节速度与末端速度的线性映射：

$$\begin{bmatrix} \mathbf{v} \\ \boldsymbol{\omega} \end{bmatrix} = \mathbf{J}(\mathbf{q}) \dot{\mathbf{q}}$$

对于旋转关节$i$，雅可比列向量为：
$$\mathbf{J}_i = \begin{bmatrix} \mathbf{z}_{i-1} \times (\mathbf{p}_n - \mathbf{p}_{i-1}) \\ \mathbf{z}_{i-1} \end{bmatrix}$$

对于移动关节$i$：
$$\mathbf{J}_i = \begin{bmatrix} \mathbf{z}_{i-1} \\ \mathbf{0} \end{bmatrix}$$

### 6.3.2 奇异性分析

奇异性发生在雅可比矩阵秩降低时，导致某些方向失去可控性。奇异性类型：

1. **边界奇异**：机械臂完全伸展或收缩
2. **内部奇异**：特定构型导致的自由度退化
3. **混合奇异**：边界和内部奇异的组合

奇异性检测指标：
- **条件数**：$\kappa(\mathbf{J}) = \sigma_{max}/\sigma_{min}$
- **可操作度**：$w = \sqrt{\det(\mathbf{J}\mathbf{J}^T)}$
- **最小奇异值**：$\sigma_{min}$直接反映接近奇异的程度

### 6.3.3 奇异性规避策略

实践中的奇异性处理方法：

1. **路径规划层面**：提前检测并绕过奇异区域
2. **控制层面**：使用阻尼伪逆或任务空间缩放
3. **机构设计**：增加冗余度或优化连杆参数

```
奇异性规避流程：
┌─────────────┐
│ 目标位姿   │
└──────┬──────┘
       ↓
┌─────────────┐
│ 奇异性检测 │ ← 计算条件数
└──────┬──────┘
       ↓
    危险？ ──Yes──→ ┌─────────────┐
       ↓           │ 路径修正   │
      No           └─────────────┘
       ↓
┌─────────────┐
│ 正常IK求解 │
└─────────────┘
```

## 6.4 冗余自由度的利用

### 6.4.1 零空间运动

对于冗余机械臂（关节数>任务维度），存在不影响末端位姿的内部运动。零空间投影算子：

$$\mathbf{N} = \mathbf{I} - \mathbf{J}^{\dagger}\mathbf{J}$$

其中$\mathbf{J}^{\dagger}$是伪逆。完整的运动控制律：

$$\dot{\mathbf{q}} = \mathbf{J}^{\dagger}\dot{\mathbf{x}} + \mathbf{N}\dot{\mathbf{q}}_0$$

第二项$\mathbf{N}\dot{\mathbf{q}}_0$在零空间内优化次要目标。

### 6.4.2 任务优先级框架

多任务场景下，按优先级分配自由度：

**严格优先级**：
$$\dot{\mathbf{q}} = \sum_{i=1}^{n} \mathbf{N}_{i-1}\mathbf{J}_i^{\dagger}(\dot{\mathbf{x}}_i - \mathbf{J}_i\sum_{j=1}^{i-1}\mathbf{N}_{j-1}\mathbf{J}_j^{\dagger}\dot{\mathbf{x}}_j)$$

其中$\mathbf{N}_i = \mathbf{N}_{i-1} - \mathbf{J}_i^{\dagger}\mathbf{J}_i\mathbf{N}_{i-1}$是递归零空间。

**软优先级（加权）**：
$$\min_{\dot{\mathbf{q}}} \sum_{i} w_i ||\mathbf{J}_i\dot{\mathbf{q}} - \dot{\mathbf{x}}_i||^2$$

通过二次规划(QP)求解，允许任务间的妥协。

### 6.4.3 次要目标优化

零空间可用于优化各种次要目标：

1. **避障**：
$$\dot{\mathbf{q}}_0 = k_{obs} \nabla_{\mathbf{q}} \sum_{i} \frac{1}{d_i^2}$$
其中$d_i$是到障碍物的距离。

2. **关节极限规避**：
$$\dot{\mathbf{q}}_0 = -k_{lim} \nabla_{\mathbf{q}} H(\mathbf{q})$$
其中$H(\mathbf{q}) = \sum_{i} \frac{(q_i - \bar{q}_i)^2}{(q_{i,max} - q_{i,min})^2}$

3. **操作性最大化**：
$$\dot{\mathbf{q}}_0 = k_{man} \nabla_{\mathbf{q}} \sqrt{\det(\mathbf{J}\mathbf{J}^T)}$$

## 6.5 工作空间分析与优化

### 6.5.1 可达空间与灵巧空间

**可达空间**：末端执行器能够到达的所有位置集合
- 通过蒙特卡洛采样或解析边界计算
- 受关节极限和连杆长度约束

**灵巧空间**：末端能以任意姿态到达的位置集合
- 通常远小于可达空间
- 对精确操作任务critical

轮足平台的独特优势在于可以通过移动底盘动态扩展工作空间：

$$W_{total} = W_{arm} \oplus W_{base}$$

其中$\oplus$表示闵可夫斯基和。

### 6.5.2 工作空间形状优化

机械臂设计阶段的参数优化：

**目标函数**：
- 最大化灵巧空间体积：$V_{dext}$
- 最大化全局条件指数：$GCI = \frac{1}{V} \int_W \frac{1}{\kappa(\mathbf{J})} dV$
- 最小化奇异区域：$V_{singular}$

**设计变量**：
- 连杆长度：$\{a_i, d_i\}$
- 关节极限：$\{q_{i,min}, q_{i,max}\}$  
- 关节配置：如7-DOF的S-R-S vs S-R-R

### 6.5.3 动态工作空间规划

轮足系统可以通过协调运动扩展工作空间：

```
静态工作空间 → 移动扩展 → 动态工作空间
    2m³           +3m²          覆盖整个房间
```

规划策略：
1. **基座最优定位**：给定任务，确定最佳站立位置
2. **在线重定位**：任务执行中动态调整基座
3. **全身协调**：同时优化基座和臂部运动

## 6.6 案例研究：Franka Emika Panda七自由度设计

Franka Emika Panda是协作机器人的典范，其7-DOF设计体现了冗余度在实际应用中的价值。

### 设计特点

1. **S-R-S构型**：肩部3DOF + 肘部1DOF + 腕部3DOF
2. **关节布局**：采用偏置设计避免机械干涉
3. **冗余度利用**：第3个关节（肘部冗余）实现零空间运动

### 运动学参数

DH参数表（简化）：
| Joint | a(m) | d(m) | α(rad) | θ range(rad) |
|-------|------|------|--------|--------------|
| 1 | 0 | 0.333 | 0 | ±2.90 |
| 2 | 0 | 0 | -π/2 | ±1.76 |
| 3 | 0 | 0.316 | π/2 | ±2.90 |
| 4 | 0.0825 | 0 | π/2 | ±3.07 |
| 5 | -0.0825 | 0.384 | -π/2 | ±2.90 |
| 6 | 0 | 0 | π/2 | ±1.66 |
| 7 | 0.088 | 0.107 | π/2 | ±2.90 |

### 冗余度应用

Panda利用冗余度实现多种功能：

1. **碰撞规避**：通过肘部运动绕过障碍物
2. **人机协作**：保持"自然"的手臂姿态
3. **奇异性规避**：通过零空间运动远离奇异构型
4. **关节限位规避**：优化关节角度分布

### 控制实现

Panda的逆运动学求解采用混合策略：
```cpp
// 伪代码
Solution computeIK(Pose target, Config current) {
    // 1. 快速解析解（忽略冗余度）
    vector<Config> analytical_solutions = analyticalIK(target);
    
    // 2. 冗余度优化
    Config best = selectBest(analytical_solutions, current);
    
    // 3. 零空间调整
    best = optimizeNullspace(best, secondary_tasks);
    
    // 4. 数值精修
    best = numericalRefinement(best, target);
    
    return best;
}
```

性能指标：
- IK求解时间：< 1ms
- 位置精度：< 0.1mm  
- 姿态精度：< 0.1°
- 奇异性规避成功率：> 99%

## 6.7 高级话题：任务优先级与零空间投影

### 6.7.1 增广雅可比方法

处理多个任务约束时，增广雅可比提供统一框架：

$$\begin{bmatrix} \dot{\mathbf{x}}_1 \\ \dot{\mathbf{x}}_2 \\ \vdots \end{bmatrix} = \begin{bmatrix} \mathbf{J}_1 \\ \mathbf{J}_2 \\ \vdots \end{bmatrix} \dot{\mathbf{q}}$$

但直接求伪逆会导致任务冲突时的不可预测行为。

### 6.7.2 递归零空间投影

严格的优先级保证通过递归投影实现：

```python
def hierarchical_ik(tasks, q_current):
    q_dot = zeros(n_joints)
    N = eye(n_joints)  # 初始零空间为全空间
    
    for task in tasks:
        J_task = task.jacobian(q_current)
        x_dot_task = task.desired_velocity()
        
        # 投影到当前可用零空间
        J_proj = J_task @ N
        
        # 计算该任务的贡献
        q_dot_task = pinv(J_proj) @ (x_dot_task - J_task @ q_dot)
        q_dot += N @ q_dot_task
        
        # 更新零空间
        N = N @ (eye(n_joints) - pinv(J_proj) @ J_proj)
    
    return q_dot
```

### 6.7.3 连续零空间转换

任务切换时的平滑过渡：

$$\mathbf{N}(t) = (1-s(t))\mathbf{N}_1 + s(t)\mathbf{N}_2$$

其中$s(t)$是光滑过渡函数（如5次多项式）。

### 6.7.4 动态任务分配

基于任务完成度和系统状态动态调整优先级：

$$w_i(t) = \begin{cases}
w_{i,max} & \text{if } e_i > e_{critical} \\
w_{i,max} \cdot \frac{e_i}{e_{critical}} & \text{otherwise}
\end{cases}$$

## 6.8 常见陷阱与错误 (Gotchas)

### 陷阱1：忽视数值精度问题

**问题**：浮点误差累积导致运动学链末端偏差
```python
# 错误示例
T = eye(4)
for i in range(n_joints):
    T = T @ compute_transform(theta[i])  # 误差累积
```

**解决**：定期重新计算完整变换，使用高精度库

### 陷阱2：奇异点附近的不稳定

**问题**：接近奇异点时微小输入产生巨大输出
```python
# 危险操作
q_dot = inv(J) @ x_dot  # 当det(J)→0时爆炸
```

**解决**：使用阻尼伪逆或切换到缩减任务空间

### 陷阱3：多解选择不一致

**问题**：IK求解器在相似输入下返回不同解，导致关节跳变

**解决**：
- 保持解的连续性跟踪
- 使用上一时刻解作为初值
- 实现解分支的显式管理

### 陷阱4：忽略关节速度/加速度限制

**问题**：IK只考虑位置，导致不可实现的运动

**解决**：在IK中集成动力学约束：
$$\dot{q}_{min} \leq \dot{\mathbf{q}} \leq \dot{q}_{max}$$

### 陷阱5：工作空间边界处理不当

**问题**：目标超出工作空间时IK发散或振荡

**解决**：
- 预先进行可达性检查
- 实现"最接近点"投影
- 提供用户反馈

### 陷阱6：实时性能瓶颈

**问题**：复杂IK算法无法满足控制周期要求

**解决**：
- 预计算和查表
- 多线程/GPU加速
- 自适应精度控制

## 6.9 最佳实践检查清单

### 设计阶段
- [ ] 工作空间是否满足任务需求？
- [ ] 是否存在不可避免的奇异构型？
- [ ] 冗余度设计是否合理？
- [ ] DH参数是否已验证无误？
- [ ] 关节极限是否考虑了机械干涉？

### 实现阶段
- [ ] IK求解器是否处理所有异常情况？
- [ ] 数值方法是否有适当的收敛准则？
- [ ] 是否实现了多解管理策略？
- [ ] 实时性能是否满足控制要求？
- [ ] 是否有奇异性检测和规避机制？

### 测试阶段
- [ ] 是否测试了工作空间边界？
- [ ] 是否验证了奇异点附近的行为？
- [ ] 是否测试了快速运动下的跟踪精度？
- [ ] 是否验证了关节极限保护？
- [ ] 是否测试了异常输入的鲁棒性？

### 部署阶段
- [ ] 是否有运动学标定程序？
- [ ] 是否记录IK失败案例？
- [ ] 是否监控计算时间？
- [ ] 是否有降级运行模式？
- [ ] 参数更新是否有版本控制？

## 本章小结

本章系统介绍了轮足机械臂的运动学理论与工程实践。关键要点包括：

1. **正运动学**是机器人控制的基础，DH参数提供了标准化的建模方法
2. **逆运动学**的求解需要权衡精度、速度和鲁棒性，混合解析-数值方法often最优
3. **雅可比矩阵**不仅用于速度映射，也是奇异性分析和冗余度利用的核心
4. **冗余自由度**通过零空间投影实现次要目标优化，显著提升系统能力
5. **工作空间分析**指导机械设计和任务规划，轮足平台的移动性提供unique优势
6. **任务优先级框架**使复杂多目标控制成为可能，但需要careful设计避免冲突

掌握这些概念和方法是实现高性能机器人控制的前提。下一章将在运动学基础上引入动力学，探讨力/力矩的传递与控制。

## 练习题

### 基础题

**题6.1** 给定3-DOF平面机械臂，连杆长度分别为$l_1=0.5m$, $l_2=0.4m$, $l_3=0.3m$。计算末端到达点$(0.8, 0.6)$的所有可能关节配置。

*Hint: 利用几何关系，先确定第三关节的可能位置*

<details>
<summary>参考答案</summary>

设第三关节位置为$(x_3, y_3)$，满足：
- 到末端距离：$(x_3-0.8)^2 + (y_3-0.6)^2 = 0.3^2$
- 到基座距离：$x_3^2 + y_3^2 \leq (l_1+l_2)^2$

求解得两组解（肘部上/下）：
- 解1：$\theta_1=45°, \theta_2=-30°, \theta_3=75°$
- 解2：$\theta_1=25°, \theta_2=40°, \theta_3=-15°$

</details>

**题6.2** 推导2-DOF机械臂的雅可比矩阵，并分析其奇异构型。

*Hint: 计算末端位置对关节角的偏导数*

<details>
<summary>参考答案</summary>

末端位置：
$$\mathbf{p} = \begin{bmatrix} l_1c_1 + l_2c_{12} \\ l_1s_1 + l_2s_{12} \end{bmatrix}$$

雅可比矩阵：
$$\mathbf{J} = \begin{bmatrix} -l_1s_1-l_2s_{12} & -l_2s_{12} \\ l_1c_1+l_2c_{12} & l_2c_{12} \end{bmatrix}$$

奇异性分析：$\det(\mathbf{J}) = l_1l_2s_2 = 0$
- 当$\theta_2 = 0°$：完全伸展
- 当$\theta_2 = 180°$：完全折叠

</details>

**题6.3** 某6-DOF机械臂采用解析IK，平均求解时间0.5ms。若切换到迭代方法（5次迭代收敛），每次迭代0.2ms，分析两种方法的适用场景。

*Hint: 考虑精度要求、实时性、开发复杂度*

<details>
<summary>参考答案</summary>

解析法优势：
- 速度快（0.5ms < 1.0ms）
- 精度高（无迭代误差）
- 适合高频控制（>1kHz）

迭代法优势：
- 通用性强（任意构型）
- 可处理冗余度
- 易于加入约束

选择建议：
- 标准6-DOF + 高实时性 → 解析法
- 7-DOF或特殊构型 → 迭代法
- 可以混合：解析法初值 + 迭代精修

</details>

### 挑战题

**题6.4** 设计7-DOF机械臂的零空间运动策略，同时优化：(1)避开障碍物 (2)远离关节极限 (3)最大化操作度。给出统一的优化框架。

*Hint: 构建加权目标函数，投影到零空间*

<details>
<summary>参考答案</summary>

统一目标函数：
$$\dot{\mathbf{q}}_0 = \alpha_1 \nabla_q f_{obs} + \alpha_2 \nabla_q f_{limit} + \alpha_3 \nabla_q f_{manip}$$

其中：
- $f_{obs} = \sum_i e^{-d_i/\sigma}$（障碍物势场）
- $f_{limit} = -\sum_i \ln\left(\frac{q_{max,i}-q_i}{q_{max,i}-q_{min,i}}\right)$（极限势场）
- $f_{manip} = \sqrt{\det(\mathbf{J}\mathbf{J}^T)}$（操作度）

权重自适应：
$$\alpha_i = \frac{w_i}{\sum_j w_j}, \quad w_i = e^{-cost_i/T}$$

温度参数$T$控制优先级软硬程度。

</details>

**题6.5** 轮足机械臂需要抓取地面上1.5m范围内的物体。机械臂工作半径1.2m，设计基座运动策略使得：(1)最小化基座运动 (2)最大化抓取成功率。

*Hint: 考虑工作空间交集和操作度分布*

<details>
<summary>参考答案</summary>

最优基座位置求解：
$$\mathbf{p}_{base}^* = \arg\min_{\mathbf{p}} \left( \lambda_1 ||\mathbf{p} - \mathbf{p}_0|| + \lambda_2 \int_{\Omega} \frac{1}{w(\mathbf{x}, \mathbf{p})} d\mathbf{x} \right)$$

其中$w(\mathbf{x}, \mathbf{p})$是位置$\mathbf{p}$时点$\mathbf{x}$的操作度。

实施策略：
1. 离散化目标区域为网格
2. 预计算每个网格点的最优基座位置
3. 在线查表 + 局部优化
4. 考虑移动成本的滞后切换

</details>

**题6.6** 某机械臂在装配任务中需要同时满足：位置精度0.1mm、力控制5N精度、避免奇异点（条件数<100）。设计一个统一的QP控制框架。

*Hint: 将所有约束转化为线性不等式*

<details>
<summary>参考答案</summary>

QP问题formulation：
$$\min_{\dot{\mathbf{q}}} \frac{1}{2}\dot{\mathbf{q}}^T\mathbf{H}\dot{\mathbf{q}} + \mathbf{f}^T\dot{\mathbf{q}}$$

约束条件：
1. 位置跟踪：$||\mathbf{J}_p\dot{\mathbf{q}} - \dot{\mathbf{x}}_d|| \leq \epsilon_p$
2. 力控制：$||\mathbf{J}_f^T\boldsymbol{\tau} - \mathbf{F}_d|| \leq \epsilon_f$
3. 奇异性：通过约束最小奇异值 $\sigma_{min}(\mathbf{J}) \geq \sigma_{threshold}$
4. 关节极限：$\mathbf{q}_{min} + \Delta t\dot{\mathbf{q}}_{min} \leq \mathbf{q} + \Delta t\dot{\mathbf{q}} \leq \mathbf{q}_{max} - \Delta t\dot{\mathbf{q}}_{max}$

线性化处理奇异性约束，使用SQP迭代求解。

</details>

**题6.7（开放题）** 下一代轮足机械臂将集成变刚度执行器。分析这对运动学控制的影响，提出相应的控制架构修改。

*Hint: 考虑刚度对精度、稳定性、安全性的影响*

<details>
<summary>参考答案</summary>

变刚度带来的挑战：
1. 位置精度依赖于负载和刚度
2. 动态响应随刚度变化
3. 需要同时控制位置和刚度

修改的控制架构：
1. **扩展状态空间**：$[\mathbf{q}, \mathbf{k}]$（位置+刚度）
2. **增广雅可比**：包含刚度对末端影响
3. **任务分配**：
   - 自由空间：低刚度快速运动
   - 接触：高刚度精确控制
   - 人机协作：可变刚度保证安全
4. **优化目标**：最小化能耗同时满足精度

实现建议：
- 在线刚度辨识
- 基于学习的刚度策略
- 考虑刚度切换的瞬态响应

</details>

**题6.8（思考题）** 讨论群机器人协作场景下的运动学协调问题。多个轮足机械臂如何协同搬运大型物体？

*Hint: 考虑闭链约束、负载分配、通信延迟*

<details>
<summary>参考答案</summary>

关键技术挑战：

1. **闭链运动学**：
   - 多臂抓取形成闭链
   - 需满足几何一致性约束
   - 虚拟关节模型

2. **负载优化分配**：
   $$\min \sum_i ||\mathbf{F}_i||^2 \text{ s.t. } \sum_i \mathbf{F}_i = \mathbf{F}_{ext}$$

3. **分布式协调**：
   - Leader-Follower架构
   - 一致性协议
   - 时延补偿

4. **容错机制**：
   - 单机故障的负载重分配
   - 降级运行模式

实施框架：
- 中央规划 + 局部控制
- 基于事件的重规划
- 力/位混合控制确保内力不过大

</details>
