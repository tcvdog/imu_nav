#!/usr/bin/env python3
"""
惯性导航小模型 — IMU → 位置估计
输入：加速度+陀螺仪 (6轴)，2秒窗口 (200帧)
输出：二维位移 (Δx, Δy)
模型：~500K 参数，4GB 显卡足够
"""

import torch
import torch.nn as nn
import numpy as np

# ── 模型 ─────────────────────────────────────────────────
class InertialNavNet(nn.Module):
    """超轻量惯性导航模型 (~500K 参数)"""
    def __init__(self, input_dim=6, hidden=128):
        super().__init__()
        # 1D CNN 提取局部特征
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=5),  # 6→32通道
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),  # 降采样到32帧
        )
        # LSTM 捕捉时序依赖
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        # 输出头
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Δx, Δy
        )

    def forward(self, x):
        # x: (batch, seq_len, 6) → (batch, 6, seq_len)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)        # (batch, 64, 32)
        x = x.permute(0, 2, 1) # (batch, 32, 64)
        x, _ = self.lstm(x)
        x = x[:, -1, :]        # 取最后一步
        return self.fc(x)


# ── 训练流程 ───────────────────────────────────────────
def train_model(csv_path, epochs=50):
    """
    数据格式 (CSV):
    timestamp,ax,ay,az,gx,gy,gz,delta_x,delta_y
    """
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader

    class IMUDataset(Dataset):
        def __init__(self, csv_path, window=200):
            df = pd.read_csv(csv_path)
            self.imu = df[['ax','ay','az','gx','gy','gz']].values
            self.target = df[['delta_x','delta_y']].values
            self.window = window

        def __len__(self):
            return len(self.imu) - self.window

        def __getitem__(self, idx):
            x = self.imu[idx:idx+self.window]
            y = self.target[idx+self.window]
            return torch.FloatTensor(x), torch.FloatTensor(y)

    dataset = IMUDataset(csv_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = InertialNavNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            pred = model(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss = {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), "inertial_nav.pth")
    print("✅ 模型已保存: inertial_nav.pth")
    return model


if __name__ == "__main__":
    # 测试模型参数量
    model = InertialNavNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"📊 模型参数量: {params:,} ({params/1e6:.2f}M)")
    print(f"💾 期望显存占用: ~{params*4/1024/1024:.0f}MB (float32)")
    print()
    print("用法: python3 train_imu.py 采集的数据.csv")
