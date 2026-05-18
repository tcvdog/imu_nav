# 🧭 IMU 惯性导航 — 手机传感器 → 实时位置估计

基于手机加速度计 + 陀螺仪 + 磁力计 + GPS，训练轻量 CNN+LSTM 模型，实现无 GPS 环境下的实时位置跟踪。

## 项目概述

| 组件 | 说明 |
|------|------|
| **Android App** | Kotlin 采集器 + 导航推理，PyTorch Android 部署 |
| **训练管线** | Python, Conv1D+LSTM, 119K 参数, TorchScript 导出 |
| **数据格式** | `timestamp,ax,ay,az,gx,gy,gz,mx,my,mz,lat,lng,alt,speed,bearing` |
| **模型输入** | 400 帧加速度/陀螺仪（≈4秒窗口） |
| **模型输出** | 4秒内的位移 (Δx, Δy)，累计得总位置 |

## 架构

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐
│  手机传感器采集   │────▶│  Conv1D+LSTM  │────▶│  实时位置显示     │
│ (Accel/Gyro/Mag) │     │  119K参数     │     │  (累计位移)      │
└─────────────────┘     └──────────────┘     └──────────────────┘
       │                       ▲
       │  GPS (训练时)          │  TorchScript 推理
       ▼                       │
┌─────────────────┐     ┌──────┴───────┐
│  训练数据 CSV    │────▶│  train_imu.py │
│  IMU+GPS 真值   │     │  学习IMU→位移  │
└─────────────────┘     └──────────────┘
```

## 手机 App 功能

### 记录模式
- 采集加速度 + 陀螺仪 + 磁力计 + GPS 到 CSV 文件
- 前台服务保活 + 通知栏状态
- 每秒更新预览数据
- 数据存到 `Android/data/com.imucollector/files/Documents/IMUData/`

### 导航模式
- 加载 `imu_model.pt`（TorchScript）
- 400 帧滑动窗口推理（≈4秒），每 80 帧推理一次
- EMA 低通滤波（α=0.3）减少传感器噪声
- 输出死区阈值（<0.01m 忽略），静止不漂移
- 累计位移显示当前位置和总路程

### 手机端处理
| 处理 | 说明 |
|------|------|
| EMA 滤波 | `filt = α * raw + (1-α) * filt`，α=0.3 |
| 死区阈值 | 模型输出 <0.01m 忽略 |
| 传感器缓存 | 450 帧环形缓冲区 |
| 推理频率 | 每 80 帧（≈0.8秒）一次 |

## 训练管线

### 数据预处理
```bash
# 低通滤波 → 去重力 → GPS 差分 → 训练样本
python3 preprocess_imu.py imu_data.csv
```

### 训练
```bash
python3 train_imu.py data/imu_xxx.csv
```

### 模型导出
```bash
python3 export_android.py   # → imu_model.pt (TorchScript, 485KB)
```

### 模型架构
```python
IMUNet(
  (cnn): Sequential(
    Conv1d(6, 32, kernel=5) → ReLU
    Conv1d(32, 64, kernel=5) → ReLU
    AdaptiveAvgPool1d(32)
  )
  (lstm): LSTM(64 → 128)
  (fc): Linear(128→64) → ReLU → Linear(64→2)
)
# 参数量: 119,010 (0.5MB)
```

### 训练数据
- GPS 只在位置变化 >0.5m 时产生有效标签
- 窗口长度：400 帧（≈4秒），确保位移信号 > GPS 噪声
- 最佳模型误差：**3.5m**（4秒窗口位移预测）

## 项目文件

```
imu_nav/
├── train_imu.py          # 模型定义 + 训练
├── export_android.py     # TorchScript 导出
├── preprocess_imu.py     # 数据预处理（滤波+去重力）
├── run_model.py          # 模型推理 + 轨迹可视化
├── imu_model.pt          # 手机端 TorchScript 模型（485KB）
├── imu_model.pth         # PyTorch 权重文件
├── data/                 # CSV 训练数据
├── docs/                 # 说明文档
├── requirements.txt      # Python 依赖
└── README.md

IMUCollector/             # Android 采集+导航 App
├── app/src/main/java/com/imucollector/
│   ├── MainActivity.kt       # 主界面（记录/导航/停止按钮）
│   └── RecordingService.kt   # 后台服务（采集+推理）
├── app/build.gradle          # 依赖配置
└── app/src/main/res/         # UI 布局
```

## 依赖

### Python 训练环境
```bash
torch>=2.0.0          # 模型训练 + TorchScript 导出
numpy>=1.21.0         # 数据处理
pandas>=1.3.0         # CSV 读取
scipy>=1.7.0          # 低通滤波（butter+filfilt）
matplotlib>=3.4.0     # 轨迹可视化
jieba>=0.42.0         # 中文分词（可选，用于多音字检测）
geopy                 # GPS 坐标→米转换
```

### Android 构建环境
- Android Studio (API 34, Kotlin 1.9.20)
- PyTorch Android 1.12.2 (`org.pytorch:pytorch_android:1.12.2`)
- Jetpack AppCompat 1.6.1
- Material Design 1.11.0
- NDK: arm64-v8a ABI
- 编译 SDK 34, 最低 SDK 26

### Android 手机要求
- Android 8.0+ (API 26+)
- 传感器：加速度计（必需）、陀螺仪（推荐）、磁力计
- 定位：GPS + 网络定位
- 存储：64MB APK 安装空间 + 数据文件空间

## 部署

```bash
# 1. 推 APK 到手机
adb push IMUCollector.apk /sdcard/Download/

# 2. 手动安装（MIUI 等系统需手动确认）
# 3. 推模型到内部目录
adb push imu_model.pt /data/local/tmp/
adb shell "run-as com.imucollector sh -c 'cat /data/local/tmp/imu_model.pt > /data/data/com.imucollector/files/imu_model.pt && chmod 644 /data/data/com.imucollector/files/imu_model.pt'"
```

## 版本历史

| 版本 | 说明 |
|------|------|
| v1.0 | 基础 IMU 采集器，仅记录数据 |
| v2.0 | 添加导航模式，PyTorch Android 推理，80帧窗口 |
| v2.1 | 修复闪退（libpytorch_jni.so 缺失），400帧长窗口，EMA滤波，死区阈值，双GPS定位 |

## 已知限制
- 模型依赖训练数据的运动模式，不同运动模式需重新训练
- GPS 噪声较大（~5m），短窗口位移信号弱
- 无陀螺仪的手机只能靠加速度+磁力计定位，精度受限
- 目前仅 arm64-v8a 架构（64位 ARM 设备）

## 下一步
- [ ] 采集更多带陀螺仪的行走数据重新训练
- [ ] 增加 Kalman 滤波融合 GPS+IMU
- [ ] 轨迹实时绘制
- [ ] 鸿蒙 HarmonyOS 版本
