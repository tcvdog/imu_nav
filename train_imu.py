#!/usr/bin/env python3
"""快速训练 — IMU+GPS → 位移估计"""
import csv, math, torch, sys
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ── 模型 ────────────────────────────────────────────
class IMUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(6, 32, 5), nn.ReLU(),
            nn.Conv1d(32, 64, 5), nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x).permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

# ── 数据 ────────────────────────────────────────────
def load_data(csv_path, window=400, stride=80):
    """读取CSV，GPS只在变化时产生有效标签"""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    'ax': float(r['ax']), 'ay': float(r['ay']), 'az': float(r['az']),
                    'gx': float(r['gx']), 'gy': float(r['gy']), 'gz': float(r['gz']),
                    'lat': float(r['lat']), 'lng': float(r['lng']),
                })
            except: pass

    print(f"📊 总采样: {len(rows)}")

    # 过滤掉 GPS=0 的数据
    valid = [r for r in rows if r['lat'] != 0 and r['lng'] != 0]
    print(f"📡 GPS有效: {len(valid)}")

    # GPS 经纬度→米
    lat0, lng0 = valid[0]['lat'], valid[0]['lng']
    for r in valid:
        r['x'] = (r['lng'] - lng0) * 111320 * math.cos(math.radians(lat0))
        r['y'] = (r['lat'] - lat0) * 110540

    # 找到 GPS 变化的位置（真正的位移事件）
    print("🔍 检测 GPS 变化点...")
    gps_events = []  # (frame_idx, dx, dy)
    prev_x, prev_y = valid[0]['x'], valid[0]['y']
    event_count = 0
    for i in range(1, len(valid)):
        dx = valid[i]['x'] - prev_x
        dy = valid[i]['y'] - prev_y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 0.5:  # 移动超过0.5米才算
            gps_events.append((i, valid[i]['x'] - prev_x, valid[i]['y'] - prev_y))
            prev_x, prev_y = valid[i]['x'], valid[i]['y']
            event_count += 1

    print(f"📍 GPS位移事件: {event_count} 次 (过滤 >0.5m)")

    # 生成训练样本：用 GPS 事件前的 IMU 窗口预测位移
    X, Y = [], []
    for idx, dx, dy in gps_events:
        start = max(0, idx - window)
        if idx - start == window:  # 正好 window 帧
            imu = [[v['ax'], v['ay'], v['az'], v['gx'], v['gy'], v['gz']]
                   for v in valid[start:idx]]
            X.append(imu)
            Y.append([dx, dy])

    print(f"🎯 训练样本: {len(X)} (窗口{window}, GPS事件{event_count})")

    # 如果样本太少，补充窗口滑动采样
    if len(X) < 50:
        print("⚠️ GPS事件太少，补充滑动窗口样本...")
        for i in range(0, len(valid) - window - 1, stride):
            imu = [[v['ax'], v['ay'], v['az'], v['gx'], v['gy'], v['gz']]
                   for v in valid[i:i+window]]
            dx = valid[i+window]['x'] - valid[i]['x']
            dy = valid[i+window]['y'] - valid[i]['y']
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0.5:  # 只保留有实际位移的样本
                X.append(imu)
                Y.append([dx, dy])

    print(f"📊 最终训练样本: {len(X)}")

    if len(X) == 0:
        print("❌ 无有效训练样本！")
        return torch.FloatTensor([]), torch.FloatTensor([])

    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y)
    return X, Y

# ── 训练 ────────────────────────────────────────────
def train(X, Y, epochs=50):
    dataset = [(X[i], Y[i]) for i in range(len(X))]
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = IMUNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数: {params:,} ({params*4/1024/1024:.1f}MB)")

    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()

    split = int(len(X) * 0.8)
    X_train, X_test, Y_train, Y_test = X[:split], X[split:], Y[:split], Y[split:]

    best_err = float('inf')
    for epoch in range(epochs):
        model.train()
        total = 0
        for i in range(0, len(X_train), 32):
            xb, yb = X_train[i:i+32], Y_train[i:i+32]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        # 测试
        model.eval()
        with torch.no_grad():
            pred = model(X_test)
            test_loss = loss_fn(pred, Y_test).item()
            err = torch.sqrt(torch.mean((pred - Y_test)**2)).item()
            if err < best_err:
                best_err = err
                torch.save(model.state_dict(), "imu_model_best.pth")

        if epoch % 5 == 0 or epoch == epochs-1:
            print(f"  Epoch {epoch+1:2d}: train={total/max(len(X_train)//32,1):.6f}  test={test_loss:.6f}  avg_err={err:.3f}m")

    print(f"\n✅ 最佳模型误差: {best_err:.3f}m")
    # 加载最佳模型
    model.load_state_dict(torch.load("imu_model_best.pth"))
    torch.save(model.state_dict(), "imu_model.pth")
    print(f"✅ 模型已保存: imu_model.pth")
    return model

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/imu_20260518_205212.csv"
    X, Y = load_data(path, window=400, stride=80)
    if len(X) > 10:
        train(X, Y, epochs=50)
    else:
        print("⚠️ 样本太少，检查数据")
