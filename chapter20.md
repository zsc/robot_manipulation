# 第20章：基于模型的规划与控制

本章深入探讨如何将学习得到的世界模型用于机器人的规划与控制。我们将介绍模型预测控制(MPC)在学习系统中的应用，各种规划算法的原理与实现，Dreamer系列方法的演进历程，以及视频预测与动作规划的融合。通过本章学习，读者将掌握如何利用世界模型进行长期规划，实现样本高效的机器人控制。

## 20.1 模型预测控制在学习系统中的应用

### 20.1.1 MPC基础回顾

模型预测控制是一种基于模型的优化控制方法，其核心思想是利用系统的动力学模型预测未来状态，并通过求解有限时域优化问题得到控制序列。在每个时间步，MPC求解以下优化问题：

$$\min_{u_{t:t+H-1}} \sum_{k=0}^{H-1} c(x_{t+k}, u_{t+k}) + c_f(x_{t+H})$$

其中：
- $H$ 是预测时域
- $c(x, u)$ 是阶段成本函数
- $c_f(x)$ 是终端成本
- 约束条件：$x_{t+k+1} = f(x_{t+k}, u_{t+k})$（动力学约束）

### 20.1.2 学习型MPC的挑战

将MPC应用于学习系统面临独特挑战：

1. **模型不确定性**：学习得到的模型存在预测误差，尤其在数据稀疏区域
2. **计算复杂度**：神经网络模型的非凸性使优化求解困难
3. **长期预测退化**：误差随预测步数累积，导致长期规划不可靠
4. **分布偏移**：执行过程中可能遇到训练分布外的状态

### 20.1.3 鲁棒MPC设计

为应对模型不确定性，鲁棒MPC引入了几种策略：

**1. 管道MPC (Tube MPC)**
构建状态管道来包含所有可能的轨迹：
$$\mathcal{X}_k = \{x : \|x - \bar{x}_k\| \leq \epsilon_k\}$$

其中$\epsilon_k$根据模型不确定性估计确定。

**2. 随机MPC (Stochastic MPC)**
将不确定性建模为概率分布，优化期望成本：
$$\min_{\pi} \mathbb{E}_{p(x_{t+1}|x_t,u_t)}\left[\sum_{k=0}^{H-1} c(x_{t+k}, \pi(x_{t+k}))\right]$$

**3. 分布鲁棒MPC**
考虑最坏情况下的分布：
$$\min_{u} \max_{p \in \mathcal{P}} \mathbb{E}_p[J(x, u)]$$

其中$\mathcal{P}$是包含真实分布的不确定集。

### 20.1.4 神经网络MPC实现

使用神经网络作为动力学模型时，MPC优化可通过以下方法求解：

**1. 基于梯度的优化**
利用自动微分计算成本对控制的梯度：
```
for iteration in range(max_iters):
    states = rollout(x0, controls, dynamics_model)
    cost = compute_cost(states, controls)
    grad = autograd(cost, controls)
    controls = controls - lr * grad
```

**2. 采样优化方法**
- 交叉熵方法(CEM)：迭代采样和精英选择
- MPPI：基于路径积分的重要性采样
- 随机打靶法：并行评估多个轨迹

## 20.2 规划算法详解

### 20.2.1 交叉熵方法(CEM)

CEM是一种基于采样的优化算法，特别适合处理非凸优化问题。算法流程：

1. **初始化分布**：$\mathcal{N}(\mu_0, \Sigma_0)$
2. **采样动作序列**：$U^{(i)} \sim \mathcal{N}(\mu, \Sigma)$
3. **评估轨迹**：计算每个样本的累积奖励
4. **选择精英样本**：保留top-K个最优轨迹
5. **更新分布**：
   $$\mu_{new} = \frac{1}{K}\sum_{i \in \text{elite}} U^{(i)}$$
   $$\Sigma_{new} = \frac{1}{K}\sum_{i \in \text{elite}} (U^{(i)} - \mu_{new})(U^{(i)} - \mu_{new})^T$$

**CEM的关键超参数**：
- 样本数N：典型值100-1000
- 精英比例：通常10-20%
- 迭代次数：3-10次
- 平滑系数：$\mu = \alpha \mu_{new} + (1-\alpha)\mu_{old}$

### 20.2.2 模型预测路径积分控制(MPPI)

MPPI基于路径积分控制理论，将最优控制问题转化为期望估计：

**理论基础**：
最优控制的路径积分表示：
$$u^* = u_0 + \frac{\mathbb{E}[\epsilon e^{-\frac{1}{\lambda}S(\tau)}]}{\mathbb{E}[e^{-\frac{1}{\lambda}S(\tau)}]}$$

其中：
- $\epsilon$是控制噪声
- $S(\tau)$是轨迹成本
- $\lambda$是温度参数

**MPPI算法**：
```
1. 从当前控制序列添加噪声：u_k = u_nom + ε_k
2. 前向仿真N条轨迹
3. 计算轨迹成本S_i
4. 计算权重：w_i = exp(-S_i/λ) / Σ_j exp(-S_j/λ)
5. 更新控制：u_new = Σ_i w_i * u_i
```

**MPPI vs CEM对比**：
- MPPI保留所有样本信息，CEM只用精英样本
- MPPI适合连续控制，CEM更通用
- MPPI收敛更平滑，CEM可能更快找到局部最优

### 20.2.3 迭代线性二次高斯(iLQG)

iLQG通过局部线性化和二次近似求解非线性最优控制：

**前向传播**：
沿着标称轨迹积分动力学：
$$x_{k+1} = f(x_k, u_k)$$

**反向传播**：
计算值函数的二次近似：
$$V(x) = \frac{1}{2}x^T V_{xx} x + V_x^T x + V_0$$

**动力学线性化**：
$$\delta x_{k+1} = A_k \delta x_k + B_k \delta u_k$$
其中：
$$A_k = \frac{\partial f}{\partial x}\bigg|_{x_k, u_k}, \quad B_k = \frac{\partial f}{\partial u}\bigg|_{x_k, u_k}$$

**控制更新**：
$$\delta u_k^* = -K_k \delta x_k - k_k$$
其中：
$$K_k = (Q_{uu} + B_k^T V_{xx}^{k+1} B_k)^{-1} B_k^T V_{xx}^{k+1} A_k$$
$$k_k = (Q_{uu} + B_k^T V_{xx}^{k+1} B_k)^{-1}(Q_u + B_k^T V_x^{k+1})$$

**正则化技巧**：
- Levenberg-Marquardt正则化：$Q_{uu} \leftarrow Q_{uu} + \mu I$
- 线搜索：$u_{new} = u_{old} + \alpha \delta u$
- 信赖域约束：$\|\delta u\| \leq \Delta$

## 20.3 Dreamer系列方法演进

### 20.3.1 Dreamer v1：开创性架构

Dreamer v1引入了基于隐状态的世界模型学习：

**核心组件**：
1. **表示模型**：$p(s_t | s_{t-1}, a_{t-1}, o_t)$
2. **转移模型**：$p(s_t | s_{t-1}, a_{t-1})$
3. **观测模型**：$p(o_t | s_t)$
4. **奖励模型**：$p(r_t | s_t)$

**训练目标**：
变分下界(ELBO)：
$$\mathcal{L} = \sum_t \left( \mathbb{E}_q[\log p(o_t | s_t) + \log p(r_t | s_t)] - \beta \text{KL}[q(s_t | \cdot) \| p(s_t | \cdot)] \right)$$

**策略学习**：
在想象轨迹上使用actor-critic：
$$J(\pi) = \mathbb{E}_{\pi, p}\left[\sum_{t=0}^{H} \gamma^t r_t\right]$$

### 20.3.2 Dreamer v2：关键改进

**主要创新**：
1. **离散隐状态**：使用categorical分布代替高斯分布
2. **KL平衡**：动态调整KL项的梯度权重
3. **改进的值函数学习**：使用λ-return目标

**离散表示的优势**：
- 更好的多模态建模能力
- 防止后验崩塌
- 更稳定的训练

**KL平衡策略**：
$$\mathcal{L}_{KL} = \alpha \cdot \text{sg}(\text{KL}[q \| p]) + (1-\alpha) \cdot \text{KL}[\text{sg}(q) \| p]$$

其中sg表示停止梯度，$\alpha$控制平衡。

### 20.3.3 Dreamer v3：规模化与简化

**架构简化**：
- 统一的Transformer骨干网络
- Symlog变换处理不同尺度的奖励
- 简化的超参数设置

**Symlog变换**：
$$\text{symlog}(x) = \text{sign}(x) \cdot \ln(|x| + 1)$$

这允许模型处理从Atari（奖励范围[-1, 1]）到DMLab（奖励范围可达数千）的不同环境。

**关键改进**：
1. **自由比特**：防止KL项过度正则化
2. **层归一化**：提高训练稳定性
3. **EMA目标网络**：稳定值函数学习

**训练流程优化**：
```python
# 并行数据收集与模型训练
while not done:
    # 环境交互（异步）
    with env_workers:
        collect_experience()
    
    # 模型训练
    for _ in range(model_train_steps):
        batch = sample_batch()
        update_world_model(batch)
    
    # 策略改进
    for _ in range(actor_train_steps):
        imagine_trajectories()
        update_actor_critic()
```

## 20.4 视频预测与动作规划的结合

### 20.4.1 视频预测模型架构

视频预测为机器人提供了直观的未来状态表示。主要架构包括：

**1. 确定性预测模型**
基于卷积LSTM或3D卷积的架构：
$$\hat{I}_{t+1} = f_\theta(I_{t-k:t}, a_{t-k:t})$$

**2. 随机视频预测**
引入隐变量建模不确定性：
$$p(I_{t+1} | I_{t}, a_t) = \int p(I_{t+1} | z_t, I_t, a_t) p(z_t | I_t) dz_t$$

常用架构：
- SV2P (Stochastic Variational Video Prediction)
- SVG (Stochastic Video Generation)
- SAVP (Stochastic Adversarial Video Prediction)

### 20.4.2 像素级MPC

直接在像素空间进行规划的挑战与解决方案：

**视觉前瞻控制(Visual Foresight)**：
1. 使用视频预测模型预测未来帧序列
2. 定义像素级成本函数（如目标像素距离）
3. 使用CEM优化动作序列

**成本函数设计**：
- 特征距离：$c = \|\phi(I_{pred}) - \phi(I_{goal})\|^2$
- 关键点跟踪：$c = \sum_i \|kp_i^{pred} - kp_i^{goal}\|^2$
- 学习型成本：$c = f_\psi(I_{pred}, I_{goal})$

**计算优化技巧**：
```python
# 批量视频预测
def batch_video_prediction(model, init_frames, action_sequences):
    # action_sequences: [N_samples, T, action_dim]
    # 并行预测N个动作序列的结果
    predictions = model(
        init_frames.repeat(N_samples, 1, 1, 1),
        action_sequences
    )
    return predictions  # [N_samples, T, H, W, C]
```

### 20.4.3 隐空间规划

在学习的隐表示中进行规划可以提高效率：

**World Models架构**：
1. **视觉编码器**：$z_t = \text{encode}(I_t)$
2. **隐状态动力学**：$h_{t+1} = f(h_t, z_t, a_t)$
3. **解码器**：$\hat{I}_t = \text{decode}(h_t, z_t)$

**规划流程**：
```
1. 编码当前观测：z_0 = encode(I_0)
2. 在隐空间rollout：
   for t in range(H):
       h_{t+1} = dynamics(h_t, a_t)
       r_t = reward_model(h_t)
3. 优化动作序列最大化累积奖励
```

**隐空间的优势**：
- 降维：从高维像素到紧凑表示
- 平滑性：隐空间通常更平滑，利于优化
- 语义性：学习的表示捕获任务相关特征

### 20.4.4 层次化规划

结合高层符号规划和低层连续控制：

**双层架构**：
```
高层规划器 (离散/符号)
    ↓ 子目标
低层控制器 (连续/反应式)
    ↓ 动作
机器人执行
```

**子目标生成方法**：
1. **关键帧提取**：从演示中学习关键状态
2. **可达性引导**：基于学习的可达性模型
3. **信息增益**：选择最大化信息的中间目标

**时间抽象**：
不同层次使用不同时间尺度：
- 高层：每k步规划一次
- 低层：每步执行
- 优势：减少规划复杂度，提高长期规划能力

## 20.5 在线适应与元学习

### 20.5.1 模型适应策略

机器人部署时需要适应新环境和任务：

**1. 在线模型更新**
持续收集数据更新世界模型：
$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}(D_{new} \cup D_{replay})$$

关键考虑：
- 灾难性遗忘：使用经验回放或EWC
- 分布偏移检测：监控预测误差
- 选择性更新：只更新不确定的组件

**2. 残差模型学习**
学习标称模型的修正项：
$$f_{true}(x, u) = f_{nominal}(x, u) + f_{residual}(x, u)$$

优势：
- 保留先验知识
- 快速适应局部变化
- 稳定性更好

### 20.5.2 基于梯度的元学习(MAML)

使模型能够快速适应新任务：

**MAML目标**：
$$\min_\theta \sum_{\mathcal{T}_i} \mathcal{L}_{\mathcal{T}_i}(\theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta))$$

**机器人应用中的MAML**：
```python
def maml_update(model, tasks, inner_lr, outer_lr):
    meta_loss = 0
    for task in tasks:
        # 内循环：任务特定适应
        task_model = copy(model)
        for _ in range(inner_steps):
            loss = compute_loss(task_model, task.support_data)
            task_model = task_model - inner_lr * grad(loss)
        
        # 外循环：元参数更新
        meta_loss += compute_loss(task_model, task.query_data)
    
    # 更新元参数
    model = model - outer_lr * grad(meta_loss)
    return model
```

### 20.5.3 上下文条件模型

通过上下文编码实现快速适应：

**架构设计**：
$$\pi(a|s) = f_\theta(s, c_\tau)$$

其中上下文$c_\tau$编码任务特定信息：
$$c_\tau = g_\phi(\{(s_i, a_i, r_i)\}_{i=1}^k)$$

**上下文推断方法**：
1. **递归编码**：使用RNN处理历史轨迹
2. **注意力聚合**：Transformer编码交互历史
3. **对比学习**：学习任务判别性表示

### 20.5.4 测试时适应(TTA)

在部署时进行快速适应：

**自监督适应**：
利用一致性损失适应新环境：
$$\mathcal{L}_{consist} = \|f(x) - f(\mathcal{A}(x))\|^2$$

其中$\mathcal{A}$是数据增强。

**伪标签方法**：
1. 使用当前模型生成预测
2. 选择高置信度预测作为伪标签
3. 在伪标签上微调模型

**主动探索**：
设计探索策略收集信息丰富的数据：
$$a^* = \arg\max_a \mathcal{I}(s_{t+1}; \theta | s_t, a)$$

## 20.6 高级主题：长期规划与组合泛化

### 20.6.1 分层世界模型

处理不同时间尺度的预测：

**多尺度架构**：
- 低层：高频、局部动力学
- 高层：低频、抽象转移

$$
\begin{aligned}
h^{low}_{t+1} &= f^{low}(h^{low}_t, a_t) \\
h^{high}_{k+1} &= f^{high}(h^{high}_k, h^{low}_{kT:(k+1)T})
\end{aligned}
$$

### 20.6.2 组合式世界模型

通过组合基本模块实现泛化：

**对象中心表示**：
将场景分解为对象及其关系：
$$s_t = \{o_i^t\}_{i=1}^N, \quad o_i = [\text{pose}, \text{shape}, \text{material}]$$

**图神经网络动力学**：
$$o_i^{t+1} = f_{node}(o_i^t, \sum_{j \in \mathcal{N}(i)} f_{edge}(o_i^t, o_j^t, r_{ij}))$$

### 20.6.3 因果推理与反事实

**结构化因果模型**：
显式建模因果关系：
$$
\begin{aligned}
\text{gripper} &\rightarrow \text{object\_pose} \\
\text{object\_pose} &\rightarrow \text{reward}
\end{aligned}
$$

**反事实推理**：
回答"如果...会怎样"的问题：
1. 腹部(Abduction)：推断隐变量
2. 行动(Action)：修改干预变量
3. 预测(Prediction)：前向推理结果

## 案例研究：Meta的JEPA世界模型

### 背景与动机

Meta AI Research提出的Joint Embedding Predictive Architecture (JEPA)代表了世界模型的新范式。与传统的像素级预测不同，JEPA在抽象表示空间进行预测，避免了不必要的细节建模。

### 架构设计

**核心组件**：
1. **编码器网络**：$s_x = f_\theta(x)$，将观测映射到表示空间
2. **预测器网络**：$\hat{s}_y = g_\phi(s_x, z)$，预测未来表示
3. **目标编码器**：$s_y = f_{\bar{\theta}}(y)$，使用EMA更新

**训练目标**：
$$\mathcal{L} = \|s_y - \hat{s}_y\|^2$$

关键创新：
- 不需要解码器重构像素
- 预测抽象表示而非原始观测
- 使用掩码策略增强泛化

### 机器人应用实例

**任务设置**：
- 环境：真实厨房场景的物体操作
- 观测：RGB-D图像
- 动作：7自由度机械臂控制

**实现细节**：
1. **数据收集**：
   - 人类演示：1000条轨迹
   - 自主探索：使用好奇心驱动收集5000条轨迹

2. **模型训练**：
   ```python
   # JEPA训练循环
   for epoch in range(num_epochs):
       for batch in dataloader:
           # 编码当前和未来观测
           s_current = encoder(batch.current_obs)
           s_future = target_encoder(batch.future_obs)
           
           # 预测未来表示
           s_pred = predictor(s_current, batch.actions)
           
           # 计算损失
           loss = mse_loss(s_pred, s_future.detach())
           
           # 更新参数
           optimizer.step(loss)
           
           # EMA更新目标编码器
           update_target_encoder(encoder, target_encoder, tau=0.99)
   ```

3. **规划与控制**：
   - 使用CEM在表示空间优化动作序列
   - 成本函数：与目标表示的距离
   - 实时频率：30Hz控制循环

### 实验结果

**定量评估**：
- 成功率：85%（vs 像素预测基线65%）
- 样本效率：减少50%的演示数据需求
- 泛化性：在新物体上成功率75%

**关键发现**：
1. 表示空间预测比像素预测更稳定
2. 掩码预训练显著提升少样本学习能力
3. 层次化表示自然涌现（物体级、场景级）

### 经验教训

1. **表示学习的重要性**：好的表示空间比精确的像素预测更有价值
2. **自监督预训练**：大规模无标签数据可以显著提升性能
3. **计算效率**：抽象表示的规划比像素级规划快10倍

## 本章小结

本章系统介绍了基于模型的规划与控制方法，涵盖了从经典MPC到现代深度学习方法的完整谱系。关键要点包括：

1. **MPC与学习的结合**：通过神经网络模型增强MPC的表达能力，同时保留其优化框架
2. **规划算法对比**：CEM适合离散/混合空间，MPPI适合连续控制，iLQG提供局部最优保证
3. **Dreamer演进**：从连续到离散表示，从复杂到简化，展示了世界模型的发展趋势
4. **视频预测的角色**：提供直观的未来预测，但计算成本高，隐空间规划是平衡点
5. **在线适应的必要性**：实际部署需要快速适应，元学习和测试时适应是关键技术

**核心公式回顾**：
- MPC优化：$\min_{u} \sum_{t} c(x_t, u_t)$ s.t. $x_{t+1} = f(x_t, u_t)$
- MPPI更新：$u^* = \frac{\sum_i w_i u_i}{\sum_i w_i}$，$w_i = e^{-S_i/\lambda}$
- MAML目标：$\min_\theta \mathcal{L}(\theta - \alpha \nabla \mathcal{L}(\theta))$

## 练习题

### 基础题

**练习20.1**：推导CEM算法的收敛性条件
*提示*：考虑精英样本的分布变化

<details>
<summary>答案</summary>

CEM收敛需要：
1. 精英比例$\rho$满足：$\rho N > d$（d为动作维度）
2. 成本函数有界且Lipschitz连续
3. 初始分布覆盖最优解邻域
收敛速率：$O(1/\sqrt{N})$，其中N为样本数
</details>

**练习20.2**：比较MPPI和CEM的计算复杂度
*提示*：分析采样、评估和更新步骤

<details>
<summary>答案</summary>

设N为样本数，H为时域，D为动作维度：
- CEM：$O(K \cdot N \cdot H \cdot C_{eval})$，K为迭代次数
- MPPI：$O(N \cdot H \cdot C_{eval})$，单次迭代
MPPI通常更快但可能需要更多样本
</details>

**练习20.3**：解释Dreamer中KL平衡的作用
*提示*：考虑前向KL和反向KL的区别

<details>
<summary>答案</summary>

KL平衡解决了VAE训练中的后验崩塌问题：
- 前向KL：$D_{KL}[q||p]$鼓励q覆盖p的支撑
- 反向KL：$D_{KL}[p||q]$鼓励p匹配q的模式
平衡两者避免了过度正则化，保持表示的信息量
</details>

**练习20.4**：设计隐空间规划的成本函数
*提示*：考虑任务相关性和可微性

<details>
<summary>答案</summary>

有效的成本函数设计：
1. 目标距离：$c = \|h_t - h_{goal}\|^2$
2. 任务特定特征：$c = -\phi_{task}(h_t)$
3. 学习型成本：$c = f_\psi(h_t, \text{task\_embedding})$
关键是保证可微且与下游任务对齐
</details>

### 挑战题

**练习20.5**：设计处理部分可观测性的MPC方法
*提示*：结合信念状态和鲁棒优化

<details>
<summary>答案</summary>

部分可观测MPC设计：
1. 维护信念状态$b_t = p(s_t | o_{1:t}, a_{1:t-1})$
2. 使用粒子滤波或变分推断更新信念
3. 优化期望成本：$\min_u \mathbb{E}_{s \sim b}[\sum_t c(s_t, u_t)]$
4. 添加信息增益项鼓励探索：$c_{info} = -H(b_{t+1})$
5. 使用置信界优化worst-case性能
</details>

**练习20.6**：分析层次化规划的最优性损失
*提示*：考虑抽象导致的次优性

<details>
<summary>答案</summary>

层次化规划的次优性来源：
1. 时间抽象：高层决策频率低，错过最优时机
2. 状态抽象：信息损失导致决策偏差
3. 动作抽象：低层控制器的限制

界限分析：
设$\epsilon_h$为高层抽象误差，$\epsilon_l$为低层跟踪误差
总体次优性：$J^* - J^{hier} \leq H \cdot \epsilon_h + T \cdot \epsilon_l$
其中H为高层规划时域，T为总时间步
</details>

**练习20.7**：提出结合物理先验的世界模型架构
*提示*：考虑归纳偏置和可解释性

<details>
<summary>答案</summary>

物理信息神经网络(PINN)世界模型：
1. 结构：分离运动学和动力学模块
2. 约束：
   - 能量守恒：$\frac{d}{dt}(T + V) = P_{external}$
   - 动量守恒：在损失函数中添加软约束
3. 架构：
   ```
   观测 → 状态估计器 → [位置, 速度, 质量]
         ↓
   拉格朗日神经网络 L(q, \dot{q})
         ↓
   欧拉-拉格朗日方程 → 加速度
         ↓
   积分器 → 下一状态
   ```
4. 优势：泛化性强，样本效率高，可解释
</details>

**练习20.8**：设计在线元学习算法用于机器人适应
*提示*：结合MAML和在线学习

<details>
<summary>答案</summary>

在线元学习算法：
1. 维护任务缓冲区，存储最近K个任务经验
2. 交替执行：
   - 适应步：使用当前数据微调
   - 元更新：在缓冲区任务上执行MAML更新
3. 自适应内循环步数：
   - 根据适应损失动态调整
   - $n_{inner} = \min(n_{max}, \lceil \alpha \cdot \mathcal{L}_{adapt} \rceil)$
4. 防止遗忘：
   - 使用EWC或梯度投影
   - 保留关键任务的锚点参数
5. 实时性保证：
   - 限制每个时间步的计算预算
   - 使用近似二阶导数（如FOMAML）
</details>

## 常见陷阱与错误

### 1. 模型偏差累积
**问题**：长期预测中误差指数增长
**症状**：规划时域超过10步后性能急剧下降
**解决方案**：
- 使用短时域MPC（H=5-10）
- 引入模型不确定性估计
- 采用鲁棒MPC方法
- 定期重新规划

### 2. 分布偏移
**问题**：执行时遇到训练分布外的状态
**症状**：模型预测置信度低，控制不稳定
**解决方案**：
- 在线数据收集和模型更新
- 域随机化训练
- 使用ensemble模型检测OOD
- 设计安全回退策略

### 3. 计算延迟
**问题**：规划时间超过控制周期
**症状**：控制频率低，响应滞后
**解决方案**：
- 使用GPU并行化采样
- 预计算和缓存
- 降低规划频率，使用反应式低层控制
- 模型压缩和量化

### 4. 局部最优
**问题**：优化陷入次优解
**症状**：重复失败的动作模式
**解决方案**：
- 增加CEM/MPPI的样本数
- 使用多起点优化
- 添加探索噪声
- 结合全局和局部规划

### 5. 超参数敏感性
**问题**：性能对超参数极度敏感
**症状**：微小调整导致性能巨变
**解决方案**：
- 使用自适应超参数
- 网格搜索关键参数
- 归一化奖励和成本
- 使用相对而非绝对阈值

## 最佳实践检查清单

### 模型设计
- [ ] 选择合适的模型复杂度（避免过拟合）
- [ ] 包含不确定性估计（认知+偶然）
- [ ] 设计可解释的中间表示
- [ ] 验证模型的物理合理性
- [ ] 实现高效的推理pipeline

### 规划算法
- [ ] 根据问题特性选择算法（连续/离散，凸/非凸）
- [ ] 设置合理的规划时域（5-20步）
- [ ] 实现warm-start策略
- [ ] 添加安全约束和边界
- [ ] 监控优化收敛性

### 在线学习
- [ ] 设计增量学习策略
- [ ] 实现分布偏移检测
- [ ] 保护关键知识不被遗忘
- [ ] 限制更新速率避免不稳定
- [ ] 维护数据质量（去除异常值）

### 系统集成
- [ ] 确保实时性要求
- [ ] 实现优雅降级机制
- [ ] 添加性能监控和日志
- [ ] 设计模块化接口
- [ ] 准备故障恢复策略

### 评估验证
- [ ] 在多样化场景测试
- [ ] 评估长期性能稳定性
- [ ] 测试边界条件和异常情况
- [ ] 对比不同算法基线
- [ ] 记录失败案例并分析原因
