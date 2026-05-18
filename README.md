# IMU 惯性导航 — 手机传感器 → 位置估计

基于手机加速度计 + 陀螺仪 + GPS 数据，训练小模型实现惯性导航。

## 流程

```
手机采集 IMU+GPS → 低通滤波去颠簸 → 训练 LSTM 小模型 → 实时位置估计
```

## 数据采集

Android App: `~/桌面/IMUCollector/`

1. 手机装 IMU采集器 App
2. 点「开始」，走路/开车
3. 点「停止」，导出 CSV

数据格式: `timestamp,ax,ay,az,gx,gy,gz,lat,lng,alt,speed,bearing`

## 预处理

```bash
python3 preprocess_imu.py 采集的数据.csv
```

- 3Hz 低通滤波 → 滤除道路颠簸
- 去除重力分量
- GPS 差分 → 位移真值

## 训练

```bash
python3 train_imu.py 预处理后的数据.csv
```

模型: Conv1D + LSTM，约 12 万参数，4GB 显卡足够。

## 依赖

```bash
pip3 install torch numpy pandas scipy matplotlib jieba
sudo apt install ffmpeg
```

## 项目结构

| 文件 | 说明 |
|------|------|
| `train_imu.py` | 模型定义 + 训练 |
| `preprocess_imu.py` | 数据预处理（滤波+去重力） |
| `docs/IMU采集器.md` | Android 采集 App 说明 |
