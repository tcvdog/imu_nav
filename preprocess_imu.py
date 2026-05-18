#!/usr/bin/env python3
"""IMU 数据预处理 — 滤除颠簸 + 提取真实运动"""
import numpy as np
from scipy import signal

def preprocess_imu(csv_path, output_path="imu_clean.csv"):
    """
    1. 低通滤波 → 去掉高频颠簸
    2. 去除重力 → 提取真实加速度
    3. 积分 → 速度/位移 (用于训练标签)
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    # 采样率（从时间戳算）
    fs = 1.0 / df['timestamp'].diff().mean()  # Hz
    
    # ── 低通滤波 ─────────────────────────────────
    # 人体运动频率 < 3Hz，颠簸 > 10Hz
    cutoff = 3.0  # 3Hz 低通 → 保留走路/跑步，滤除颠簸
    b, a = signal.butter(4, cutoff / (fs/2), 'low')
    
    for axis in ['ax', 'ay', 'az']:
        df[f'{axis}_filtered'] = signal.filtfilt(b, a, df[axis].values)
    
    # ── 去除重力 ─────────────────────────────────
    # 重力在静止时是 9.8，方向垂直向下
    # 假设手机保持大致水平，重力主要在 az
    gravity = np.median(df['az_filtered'][:100])  # 取前100帧中位数
    df['az_nogravity'] = df['az_filtered'] - gravity
    
    # ── 生成标签：GPS 位移 ──────────────────────
    # GPS (lat,lng) → 米
    from geopy.distance import geodesic
    displacements = [(0.0, 0.0)]
    for i in range(1, len(df)):
        if i % 10 == 0:  # GPS 每 10 帧更新一次
            prev = (df.iloc[i-10]['lat'], df.iloc[i-10]['lng'])
            curr = (df.iloc[i]['lat'], df.iloc[i]['lng'])
            d = geodesic(prev, curr).meters
            # 近似方向（需要航向角）
            displacements.append((d, 0.0))
        else:
            displacements.append((0.0, 0.0))
    
    df['delta_x'] = [d[0] for d in displacements]
    df['delta_y'] = [d[1] for d in displacements]
    
    # ── 保存 ─────────────────────────────────────
    out_cols = ['timestamp',
                'ax_filtered', 'ay_filtered', 'az_nogravity',
                'gx', 'gy', 'gz',
                'delta_x', 'delta_y']
    df[out_cols].to_csv(output_path, index=False)
    
    print(f"✅ 预处理完成")
    print(f"   采样率: {fs:.0f}Hz")
    print(f"   滤波截止: {cutoff}Hz (低于此保留)")
    print(f"   重力补偿: {gravity:.2f} m/s²")
    print(f"   输出: {output_path}")
    return output_path


def visualize_filter_effect(csv_path):
    """画图对比原始 vs 滤波后的信号"""
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_csv(csv_path)
    t = df['timestamp'] - df['timestamp'].iloc[0]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    for i, axis in enumerate(['x', 'y', 'z']):
        ax = axes[i]
        raw = df[f'a{axis}']
        filtered = df.get(f'a{axis}_filtered', raw)
        ax.plot(t, raw, alpha=0.3, label='原始（含颠簸）')
        ax.plot(t, filtered, label='滤波后（真实运动）')
        ax.set_ylabel(f'a{axis} (m/s²)')
        ax.legend()
    
    axes[0].set_title('加速度信号对比：原始 vs 低通滤波')
    axes[-1].set_xlabel('时间 (秒)')
    plt.tight_layout()
    plt.savefig('imu_filter_comparison.png')
    print("✅ 对比图已保存: imu_filter_comparison.png")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "imu_data.csv"
    preprocess_imu(path)
