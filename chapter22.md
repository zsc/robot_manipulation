# 第22章：计算平台与操作系统

机器人系统的计算架构是实现复杂控制算法和AI模型的基础。本章深入探讨轮足机械臂机器人的计算平台选择、实时操作系统配置、中间件优化以及底层硬件接口设计。我们将从处理器架构的选择开始，逐步深入到实时性保证、中断处理和DMA优化等系统级优化技术。通过本章学习，读者将掌握如何为特定的机器人应用场景选择和配置最优的计算平台，以及如何在保证实时性的前提下最大化计算效率。

## 22.1 嵌入式处理器架构选择

### 22.1.1 ARM Cortex系列处理器

ARM Cortex处理器在机器人领域占据主导地位，其优势在于成熟的生态系统和广泛的工具链支持。对于轮足机械臂机器人，我们需要在不同层级选择合适的处理器：

**Cortex-M系列（实时控制层）**：
- Cortex-M4/M7：适用于电机控制和传感器数据采集
- 主频：100-600MHz
- 特点：DSP指令集、单精度FPU、确定性中断延迟
- 典型应用：FOC电机控制、IMU数据融合、编码器读取

**Cortex-A系列（应用处理层）**：
- Cortex-A53/A72：运行高级控制算法和视觉处理
- 主频：1.5-2.5GHz
- 特点：乱序执行、NEON SIMD、多核架构
- 典型应用：路径规划、SLAM、机器学习推理

**Cortex-R系列（安全关键层）**：
- Cortex-R5/R52：用于功能安全相关任务
- 特点：锁步核心、ECC保护、快速上下文切换
- 典型应用：紧急停止、故障检测、安全监控

处理器选择的关键考虑因素：

$$\text{Performance} = \frac{\text{IPC} \times \text{Frequency}}{\text{Power}} \times \text{Utilization}$$

其中IPC（Instructions Per Cycle）决定了处理器的效率，而功耗直接影响机器人的续航能力。

### 22.1.2 RISC-V架构的崛起

RISC-V作为开源指令集架构，在机器人领域展现出独特优势：

**架构特点**：
- 模块化ISA设计：基础整数指令集（RV32I/RV64I）+ 可选扩展
- 扩展集选择：
  - M：整数乘除法（电机控制必需）
  - F/D：单/双精度浮点（运动学计算）
  - V：向量扩展（视觉处理加速）
  - P：DSP扩展（信号处理）

**RISC-V在机器人中的优势**：
1. 定制化能力：可根据具体需求裁剪指令集
2. 无授权费用：降低BOM成本
3. 开放生态：便于学术研究和创新

**典型RISC-V处理器**：
- SiFive U74：多核应用处理器，用于高级控制
- 平头哥玄铁C906：支持RVV0.7.1向量扩展
- 芯来科技UX607：面向实时控制的MCU

### 22.1.3 异构多核架构设计

现代机器人系统采用异构计算架构以优化性能功耗比：

```
┌─────────────────────────────────────────┐
│          应用处理器集群                   │
│   ┌─────────┐  ┌─────────┐             │
│   │ A72 #0  │  │ A72 #1  │  (2.0GHz)   │
│   └─────────┘  └─────────┘             │
│   ┌─────────┐  ┌─────────┐             │
│   │ A53 #0  │  │ A53 #1  │  (1.5GHz)   │
│   └─────────┘  └─────────┘             │
└───────────┬─────────────────────────────┘
            │ AMBA AXI4
┌───────────┴─────────────────────────────┐
│          实时处理器集群                   │
│   ┌─────────┐  ┌─────────┐             │
│   │  M7 #0  │  │  M7 #1  │  (400MHz)   │
│   └─────────┘  └─────────┘             │
└───────────┬─────────────────────────────┘
            │ AHB-Lite
┌───────────┴─────────────────────────────┐
│          外设与协处理器                  │
│   GPU | NPU | DSP | ISP | Codec        │
└─────────────────────────────────────────┘
```

**任务分配策略**：
- 大核（A72）：路径规划、视觉处理、AI推理
- 小核（A53）：系统管理、通信协议栈
- 实时核（M7）：电机控制、传感器融合
- 协处理器：专用加速任务

## 22.2 专用AI加速芯片

### 22.2.1 NVIDIA Jetson系列

NVIDIA Jetson平台集成了ARM CPU和NVIDIA GPU，是机器人AI应用的主流选择：

**Jetson产品线对比**：

| 型号 | GPU | CPU | AI性能 | 功耗 | 适用场景 |
|------|-----|-----|--------|------|----------|
| Nano | 128-core Maxwell | 4x A57 | 0.5 TFLOPS | 5-10W | 入门级视觉 |
| TX2 | 256-core Pascal | 2x Denver + 4x A57 | 1.3 TFLOPS | 7.5-15W | 中型机器人 |
| Xavier NX | 384-core Volta | 6x Carmel | 21 TOPS | 10-20W | 自主导航 |
| AGX Orin | 2048-core Ampere | 12x A78AE | 275 TOPS | 15-60W | 大型机器人 |

**CUDA编程优化要点**：
```cuda
// 优化的矩阵乘法kernel示例
__global__ void matmul_tiled(float* C, float* A, float* B, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    float sum = 0.0f;
    for (int m = 0; m < N/TILE_SIZE; ++m) {
        As[ty][tx] = A[(by*TILE_SIZE + ty)*N + m*TILE_SIZE + tx];
        Bs[ty][tx] = B[(m*TILE_SIZE + ty)*N + bx*TILE_SIZE + tx];
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[(by*TILE_SIZE + ty)*N + bx*TILE_SIZE + tx] = sum;
}
```

### 22.2.2 国产AI芯片方案

**地平线征程系列**：
- 征程5：128 TOPS，BPU架构
- 特点：低延迟、高能效比
- 优势：本土支持、成本优势

**寒武纪思元系列**：
- 思元370：256 TOPS，MLU架构
- 支持PyTorch/TensorFlow直接部署
- 适合大模型推理

**算能BM1684X**：
- 32 TOPS INT8性能
- 支持多路视频解码
- 适合视觉密集型应用

### 22.2.3 推理优化技术

**量化策略**：
```python
# INT8量化示例
def quantize_model(model, calibration_data):
    # 收集激活值统计信息
    stats = collect_stats(model, calibration_data)
    
    # 计算量化参数
    for layer in model.layers:
        if isinstance(layer, nn.Conv2d):
            w_scale = compute_scale(layer.weight, stats[layer])
            layer.weight_scale = w_scale
            layer.weight_int8 = quantize(layer.weight, w_scale)
    
    return quantized_model
```

**推理延迟优化**：

$$\text{Latency} = \text{Compute} + \text{Memory} + \text{Sync}$$

优化策略：
1. 算子融合：减少内存访问
2. 批处理：提高吞吐量
3. 流水线并行：隐藏传输延迟

## 22.3 实时操作系统选择与配置

### 22.3.1 RT-Linux配置与优化

RT-Linux通过PREEMPT_RT补丁实现硬实时性能：

**内核配置关键参数**：
```bash
# 配置实时内核
CONFIG_PREEMPT_RT=y
CONFIG_HIGH_RES_TIMERS=y
CONFIG_NO_HZ_FULL=y
CONFIG_RCU_NOCB_CPU=y
CONFIG_RCU_NOCB_CPU_ALL=y
```

**CPU隔离与亲和性设置**：
```bash
# 隔离CPU核心用于实时任务
# /boot/cmdline.txt
isolcpus=2,3 nohz_full=2,3 rcu_nocbs=2,3

# 应用程序中设置CPU亲和性
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(2, &cpuset);  // 绑定到CPU 2
pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
```

**实时调度策略**：
```c
struct sched_param param;
param.sched_priority = 99;  // 最高优先级
pthread_setschedparam(thread, SCHED_FIFO, &param);

// 内存锁定防止页面交换
mlockall(MCL_CURRENT | MCL_FUTURE);
```

### 22.3.2 QNX微内核架构

QNX作为商业RTOS，在汽车和机器人领域广泛应用：

**微内核架构优势**：
- 内核最小化：仅包含调度、IPC、内存管理
- 故障隔离：驱动程序运行在用户空间
- 可靠性：单个组件崩溃不影响系统

**消息传递机制**：
```c
// QNX消息传递示例
typedef struct {
    uint16_t type;
    uint16_t subtype;
    float position[6];
    float velocity[6];
} robot_msg_t;

// 服务器端
int rcvid = MsgReceive(chid, &msg, sizeof(msg), NULL);
// 处理消息
MsgReply(rcvid, EOK, &reply, sizeof(reply));

// 客户端
int coid = ConnectAttach(0, 0, chid, _NTO_SIDE_CHANNEL, 0);
MsgSend(coid, &msg, sizeof(msg), &reply, sizeof(reply));
```

### 22.3.3 FreeRTOS在MCU中的应用

FreeRTOS适用于资源受限的实时控制器：

**任务创建与管理**：
```c
// 电机控制任务
void vMotorControlTask(void *pvParameters) {
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xPeriod = pdMS_TO_TICKS(1);  // 1kHz控制频率
    
    for(;;) {
        // 读取编码器
        read_encoders();
        // PID控制
        compute_pid();
        // 输出PWM
        update_pwm();
        // 严格周期执行
        vTaskDelayUntil(&xLastWakeTime, xPeriod);
    }
}

// 创建任务
xTaskCreate(vMotorControlTask, "MotorCtrl", 
            configMINIMAL_STACK_SIZE, NULL, 
            tskIDLE_PRIORITY + 3, NULL);
```

**中断服务程序设计**：
```c
void EXTI0_IRQHandler(void) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    
    // 清除中断标志
    EXTI->PR = EXTI_PR_PR0;
    
    // 发送信号量通知任务
    xSemaphoreGiveFromISR(xEncoderSemaphore, 
                          &xHigherPriorityTaskWoken);
    
    // 触发上下文切换
    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}
```

**内存管理策略**：
FreeRTOS提供五种内存分配方案，机器人系统通常采用heap_4或heap_5：

- heap_1：最简单，不支持释放
- heap_2：支持释放但有碎片
- heap_3：封装标准malloc/free
- heap_4：首次适配，支持碎片合并
- heap_5：支持非连续内存区域

```c
// 静态内存分配（推荐用于关键任务）
StaticTask_t xTaskBuffer;
StackType_t xStack[STACK_SIZE];

xTaskCreateStatic(vCriticalTask, "Critical", STACK_SIZE,
                  NULL, tskIDLE_PRIORITY + 4, xStack, &xTaskBuffer);
```

### 22.3.4 实时性能测量与验证

实时系统的性能指标包括延迟、抖动和最坏情况执行时间（WCET）：

**延迟测量方法**：
```c
// 使用cyclictest测量RT-Linux延迟
// 命令：cyclictest -p 99 -t 4 -n -i 1000 -l 100000

// 自定义延迟测量
struct timespec start, end;
clock_gettime(CLOCK_MONOTONIC, &start);
// 执行实时任务
do_realtime_task();
clock_gettime(CLOCK_MONOTONIC, &end);

long latency_ns = (end.tv_sec - start.tv_sec) * 1e9 + 
                  (end.tv_nsec - start.tv_nsec);
```

**抖动分析**：
抖动（Jitter）定义为实际执行时间与期望执行时间的偏差：

$$\text{Jitter} = \sigma(\Delta t_i) = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\Delta t_i - \overline{\Delta t})^2}$$

其中$\Delta t_i$是第i次执行的时间间隔。

**WCET分析工具**：
- 静态分析：aiT WCET Analyzer
- 测量基础：RapiTime
- 混合方法：OTAWA

### 22.3.5 多RTOS混合部署

现代机器人系统常采用多个RTOS协同工作：

```
┌──────────────────────────────────────┐
│     Linux (非实时域)                  │
│  - HMI、日志、网络通信                │
│  - 高级路径规划                       │
└────────────┬─────────────────────────┘
             │ Shared Memory / FIFO
┌────────────┴─────────────────────────┐
│     RT-Linux (软实时域)               │
│  - 运动控制、轨迹生成                 │
│  - 视觉处理、SLAM                    │
└────────────┬─────────────────────────┘
             │ SPI/UART
┌────────────┴─────────────────────────┐
│     FreeRTOS (硬实时域)              │
│  - 电机控制、传感器采集               │
│  - 安全监控、紧急停止                 │
└──────────────────────────────────────┘
```

**域间通信机制**：
1. 共享内存：低延迟，需要同步机制
2. 消息队列：解耦合，有序传输
3. RPC：远程过程调用，适合命令响应
4. DDS：数据分发服务，适合多对多通信

## 22.4 ROS2与DDS中间件优化

### 22.4.1 ROS2架构与DDS选择

ROS2采用DDS（Data Distribution Service）作为通信中间件，相比ROS1的自定义协议有显著优势：

**DDS实现对比**：

| DDS实现 | 特点 | 延迟 | 吞吐量 | 适用场景 |
|---------|------|------|--------|----------|
| Fast-DDS | 开源默认 | 中等 | 高 | 通用场景 |
| Cyclone DDS | 轻量级 | 低 | 中 | 嵌入式系统 |
| RTI Connext | 商业级 | 极低 | 极高 | 工业级应用 |
| GurumDDS | GPU加速 | 低 | 极高 | 视觉密集型 |

**QoS策略配置**：
```cpp
// 为不同类型的话题配置QoS
rclcpp::QoS sensor_qos(10);
sensor_qos.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
sensor_qos.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
sensor_qos.deadline(std::chrono::milliseconds(10));

rclcpp::QoS control_qos(1);
control_qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
control_qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
control_qos.history(RMW_QOS_POLICY_HISTORY_KEEP_LAST);
```

**DDS域配置与分区**：
```xml
<!-- DDS配置文件 -->
<dds>
    <profiles>
        <participant profile_name="robot_participant">
            <rtps>
                <builtin>
                    <discovery_config>
                        <discoveryProtocol>SIMPLE</discoveryProtocol>
                        <EDP>SIMPLE</EDP>
                        <domainId>42</domainId>
                    </discovery_config>
                </builtin>
                <port>
                    <domainIDGain>200</domainIDGain>
                    <participantIDGain>2</participantIDGain>
                </port>
            </rtps>
        </participant>
    </profiles>
</dds>
```

### 22.4.2 零拷贝与共享内存优化

ROS2支持通过共享内存实现零拷贝，显著降低大数据传输延迟：

**Iceoryx集成配置**：
```cpp
// 启用零拷贝发布者
auto publisher = node->create_publisher<sensor_msgs::msg::PointCloud2>(
    "pointcloud", 
    rclcpp::QoS(10).reliable(),
    rclcpp::PublisherOptions().use_intra_process_comm(true)
);

// 使用借用消息避免拷贝
auto loaned_msg = publisher->borrow_loaned_message();
// 直接操作loaned_msg的数据
fill_pointcloud_data(loaned_msg.get());
publisher->publish(std::move(loaned_msg));
```

**共享内存传输性能分析**：

传统方式延迟：
$$T_{traditional} = T_{serialize} + T_{copy} + T_{network} + T_{deserialize}$$

零拷贝延迟：
$$T_{zerocopy} = T_{pointer} + T_{sync}$$

对于1MB点云数据，延迟可从10ms降至0.1ms。

### 22.4.3 实时执行器配置

ROS2执行器（Executor）决定了回调函数的调度策略：

**静态单线程执行器（推荐用于实时系统）**：
```cpp
// 创建静态执行器，预分配所有资源
rclcpp::executors::StaticSingleThreadedExecutor executor;

// 添加节点
executor.add_node(control_node);
executor.add_node(sensor_node);

// 配置实时优先级
struct sched_param param;
param.sched_priority = 80;
pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);

// 运行执行器
executor.spin();
```

**回调组管理**：
```cpp
// 创建互斥和可重入回调组
auto mutex_group = node->create_callback_group(
    rclcpp::CallbackGroupType::MutuallyExclusive);
auto reentrant_group = node->create_callback_group(
    rclcpp::CallbackGroupType::Reentrant);

// 关键控制回调使用互斥组
auto control_timer = node->create_wall_timer(
    1ms, control_callback, mutex_group);

// 传感器回调使用可重入组
auto sensor_sub = node->create_subscription<sensor_msgs::msg::Imu>(
    "imu", 10, sensor_callback, 
    rclcpp::SubscriptionOptions().callback_group(reentrant_group));
```

### 22.4.4 网络传输优化

**UDP配置优化**：
```bash
# 增加UDP缓冲区大小
echo 26214400 > /proc/sys/net/core/rmem_max
echo 26214400 > /proc/sys/net/core/wmem_max
echo 26214400 > /proc/sys/net/core/rmem_default
echo 26214400 > /proc/sys/net/core/wmem_default

# 配置网络中断亲和性
echo 4 > /proc/irq/24/smp_affinity  # 绑定网卡中断到CPU2
```

**多播配置**：
```cpp
// 配置DDS多播地址
auto qos = rclcpp::QoS(10);
qos.liveliness(RMW_QOS_POLICY_LIVELINESS_AUTOMATIC);
qos.liveliness_lease_duration(std::chrono::milliseconds(1000));

// 使用自定义传输配置
rmw_qos_profile_t profile = rmw_qos_profile_sensor_data;
profile.avoid_ros_namespace_conventions = true;
```

### 22.4.5 诊断与性能分析

**ros2_tracing集成**：
```cpp
// 添加追踪点
#include <tracetools/tracetools.h>

void control_callback() {
    TRACEPOINT(callback_start, (void*)this);
    
    // 控制逻辑
    compute_control();
    
    TRACEPOINT(callback_end, (void*)this);
}
```

**性能指标监控**：
```cpp
class PerformanceMonitor : public rclcpp::Node {
    void timer_callback() {
        auto stats = this->get_topic_statistics();
        RCLCPP_INFO(get_logger(), 
            "Latency: %.2fms, Dropped: %lu",
            stats.mean_latency_ms, stats.dropped_messages);
    }
};
```

## 22.5 硬件抽象层设计

### 22.5.1 HAL架构原则

硬件抽象层（HAL）是机器人软件与底层硬件之间的桥梁，良好的HAL设计可以提高代码可移植性和可维护性：

**分层架构设计**：
```cpp
// HAL接口定义
class IMotorController {
public:
    virtual ~IMotorController() = default;
    virtual bool initialize(const MotorConfig& config) = 0;
    virtual bool setPosition(float position, float velocity) = 0;
    virtual bool setTorque(float torque) = 0;
    virtual MotorState getState() const = 0;
    virtual bool emergency_stop() = 0;
};

// 具体实现：CAN总线电机
class CANMotorController : public IMotorController {
private:
    int can_id_;
    CanBus* bus_;
    
public:
    bool initialize(const MotorConfig& config) override {
        can_id_ = config.can_id;
        bus_ = CanBus::getInstance(config.bus_id);
        return bus_->addDevice(can_id_, this);
    }
    
    bool setPosition(float position, float velocity) override {
        CanMessage msg;
        msg.id = can_id_;
        msg.data[0] = CMD_POSITION;
        memcpy(&msg.data[1], &position, 4);
        memcpy(&msg.data[5], &velocity, 4);
        return bus_->send(msg);
    }
};

// 具体实现：EtherCAT电机
class EtherCATMotorController : public IMotorController {
    // EtherCAT特定实现
};
```

**设备发现与枚举**：
```cpp
class HardwareManager {
private:
    std::map<std::string, std::unique_ptr<IMotorController>> motors_;
    std::map<std::string, std::unique_ptr<ISensor>> sensors_;
    
public:
    void discoverDevices() {
        // 扫描CAN总线
        auto can_devices = CanBus::scan();
        for (const auto& dev : can_devices) {
            auto motor = std::make_unique<CANMotorController>();
            if (motor->initialize(dev.config)) {
                motors_[dev.name] = std::move(motor);
            }
        }
        
        // 扫描EtherCAT
        auto ecat_devices = EtherCAT::scan();
        // ... 类似处理
    }
    
    IMotorController* getMotor(const std::string& name) {
        auto it = motors_.find(name);
        return (it != motors_.end()) ? it->second.get() : nullptr;
    }
};
```

### 22.5.2 设备驱动框架

**统一的驱动模型**：
```cpp
template<typename DeviceType>
class DeviceDriver {
protected:
    std::atomic<bool> running_{false};
    std::thread update_thread_;
    std::chrono::milliseconds update_period_;
    
public:
    virtual void start() {
        running_ = true;
        update_thread_ = std::thread([this]() {
            while (running_) {
                auto start = std::chrono::steady_clock::now();
                update();
                auto end = std::chrono::steady_clock::now();
                auto elapsed = end - start;
                if (elapsed < update_period_) {
                    std::this_thread::sleep_for(update_period_ - elapsed);
                }
            }
        });
    }
    
    virtual void stop() {
        running_ = false;
        if (update_thread_.joinable()) {
            update_thread_.join();
        }
    }
    
protected:
    virtual void update() = 0;
};
```

**热插拔支持**：
```cpp
class HotPlugManager {
private:
    std::mutex device_mutex_;
    std::vector<DeviceChangeCallback> callbacks_;
    
public:
    void monitorUSBDevices() {
        // 使用udev或类似机制监控USB设备
        struct udev_monitor* mon = udev_monitor_new_from_netlink(udev, "udev");
        udev_monitor_filter_add_match_subsystem_devtype(mon, "usb", NULL);
        udev_monitor_enable_receiving(mon);
        
        int fd = udev_monitor_get_fd(mon);
        while (true) {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(fd, &fds);
            
            if (select(fd+1, &fds, NULL, NULL, NULL) > 0) {
                struct udev_device* dev = udev_monitor_receive_device(mon);
                if (dev) {
                    handleDeviceEvent(dev);
                    udev_device_unref(dev);
                }
            }
        }
    }
    
    void handleDeviceEvent(struct udev_device* dev) {
        const char* action = udev_device_get_action(dev);
        if (strcmp(action, "add") == 0) {
            onDeviceAdded(dev);
        } else if (strcmp(action, "remove") == 0) {
            onDeviceRemoved(dev);
        }
    }
};
```

### 22.5.3 内存映射IO优化

对于高速外设，直接内存映射可以显著降低访问延迟：

```cpp
class MemoryMappedDevice {
private:
    void* base_addr_;
    size_t size_;
    int fd_;
    
public:
    bool map(const std::string& device_path, size_t size) {
        fd_ = open(device_path.c_str(), O_RDWR | O_SYNC);
        if (fd_ < 0) return false;
        
        base_addr_ = mmap(NULL, size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd_, 0);
        if (base_addr_ == MAP_FAILED) {
            close(fd_);
            return false;
        }
        
        size_ = size;
        return true;
    }
    
    template<typename T>
    void write_register(uint32_t offset, T value) {
        volatile T* reg = reinterpret_cast<T*>(
            static_cast<uint8_t*>(base_addr_) + offset);
        *reg = value;
        // 确保写入完成
        __sync_synchronize();
    }
    
    template<typename T>
    T read_register(uint32_t offset) {
        volatile T* reg = reinterpret_cast<T*>(
            static_cast<uint8_t*>(base_addr_) + offset);
        return *reg;
    }
};
```

### 22.5.4 通信总线抽象

**统一的总线接口**：
```cpp
class IBus {
public:
    virtual ~IBus() = default;
    virtual bool open(const BusConfig& config) = 0;
    virtual bool close() = 0;
    virtual bool write(uint32_t addr, const uint8_t* data, size_t len) = 0;
    virtual bool read(uint32_t addr, uint8_t* data, size_t len) = 0;
    virtual bool transaction(const Transaction& tx) = 0;
};

// SPI总线实现
class SPIBus : public IBus {
private:
    int fd_;
    spi_ioc_transfer xfer_[2];
    
public:
    bool write(uint32_t addr, const uint8_t* data, size_t len) override {
        xfer_[0].tx_buf = reinterpret_cast<uintptr_t>(&addr);
        xfer_[0].len = sizeof(addr);
        xfer_[1].tx_buf = reinterpret_cast<uintptr_t>(data);
        xfer_[1].len = len;
        
        return ioctl(fd_, SPI_IOC_MESSAGE(2), xfer_) >= 0;
    }
};

// I2C总线实现
class I2CBus : public IBus {
private:
    int fd_;
    uint8_t slave_addr_;
    
public:
    bool write(uint32_t addr, const uint8_t* data, size_t len) override {
        i2c_msg msgs[2];
        msgs[0].addr = slave_addr_;
        msgs[0].flags = 0;
        msgs[0].buf = reinterpret_cast<uint8_t*>(&addr);
        msgs[0].len = sizeof(addr);
        
        msgs[1].addr = slave_addr_;
        msgs[1].flags = 0;
        msgs[1].buf = const_cast<uint8_t*>(data);
        msgs[1].len = len;
        
        i2c_rdwr_ioctl_data msgset = {msgs, 2};
        return ioctl(fd_, I2C_RDWR, &msgset) >= 0;
    }
};
```

### 22.5.5 错误处理与恢复

**分级错误处理策略**：
```cpp
enum class ErrorSeverity {
    INFO,      // 信息性消息
    WARNING,   // 警告，系统可继续运行
    ERROR,     // 错误，功能受限
    CRITICAL,  // 严重错误，需要干预
    FATAL      // 致命错误，系统停止
};

class ErrorHandler {
private:
    struct ErrorContext {
        std::chrono::steady_clock::time_point timestamp;
        std::string component;
        ErrorSeverity severity;
        std::string message;
        std::function<bool()> recovery_action;
    };
    
    std::deque<ErrorContext> error_history_;
    std::map<std::string, int> error_counts_;
    
public:
    void handleError(const ErrorContext& ctx) {
        // 记录错误
        error_history_.push_back(ctx);
        error_counts_[ctx.component]++;
        
        // 根据严重程度采取行动
        switch (ctx.severity) {
            case ErrorSeverity::WARNING:
                logWarning(ctx);
                break;
                
            case ErrorSeverity::ERROR:
                if (error_counts_[ctx.component] > 3) {
                    // 频繁错误，尝试恢复
                    if (ctx.recovery_action && ctx.recovery_action()) {
                        error_counts_[ctx.component] = 0;
                    }
                }
                break;
                
            case ErrorSeverity::CRITICAL:
                enterSafeMode();
                notifyOperator(ctx);
                break;
                
            case ErrorSeverity::FATAL:
                emergencyStop();
                break;
        }
    }
    
    void enterSafeMode() {
        // 降低系统性能，确保安全
        setMotorTorqueLimit(0.3f);  // 限制力矩输出
        setMaxVelocity(0.5f);        // 限制速度
        disableAutonomousMode();     // 禁用自主模式
    }
};
```
