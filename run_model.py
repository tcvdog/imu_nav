#!/usr/bin/env python3
"""
使用训练好的模型进行实时位置估计
"""
import csv, math, torch, sys, os
import numpy as np

# 模型结构需要和训练时一致
class IMUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(6, 32, 5), torch.nn.ReLU(),
            torch.nn.Conv1d(32, 64, 5), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(32),
        )
        self.lstm = torch.nn.LSTM(64, 128, batch_first=True)
        self.fc = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x).permute(0, 2, 1)
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

def predict_from_csv(csv_path, model_path="imu_model.pth", window=80):
    """读取CSV，用模型逐窗口预测位置"""
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IMUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 读取数据
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    'ax': float(r['ax']),'ay': float(r['ay']),'az': float(r['az']),
                    'gx': float(r['gx']),'gy': float(r['gy']),'gz': float(r['gz']),
                })
            except: pass

    print(f"📊 数据: {len(rows)} 帧")

    # 逐窗口预测，累加位移
    x, y = 0.0, 0.0
    trajectory = [(x, y)]

    for i in range(0, len(rows) - window, 10):  # 步长10帧
        imu = [[r['ax'],r['ay'],r['az'],r['gx'],r['gy'],r['gz']]
               for r in rows[i:i+window]]
        tensor = torch.FloatTensor(imu).unsqueeze(0).to(device)
        with torch.no_grad():
            dx, dy = model(tensor)[0].cpu().numpy()
        x += dx
        y += dy
        trajectory.append((x, y))

    return trajectory


def plot_trajectory(trajectory, save_path="trajectory.png"):
    """绘制轨迹图"""
    try:
        import matplotlib.pyplot as plt
        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]

        plt.figure(figsize=(10, 8))
        # 轨迹线
        plt.plot(xs, ys, 'b-', linewidth=1, alpha=0.7, label='预测路径')
        # 起点终点
        plt.plot(xs[0], ys[0], 'go', markersize=10, label='起点')
        plt.plot(xs[-1], ys[-1], 'ro', markersize=10, label='终点')
        # 颜色渐变显示时序
        sc = plt.scatter(xs, ys, c=range(len(xs)), cmap='viridis', s=5, alpha=0.5)
        plt.colorbar(sc, label='时间 →')

        plt.axis('equal')
        plt.title(f'惯性导航轨迹 (共{len(trajectory)}步)')
        plt.xlabel('X 位移 (米)')
        plt.ylabel('Y 位移 (米)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"📈 轨迹图: {save_path}")
        plt.close()
    except ImportError:
        print("⚠️ matplotlib 未安装，跳过绘图")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 run_model.py 数据.csv [模型.pth]")
        sys.exit(1)

    csv_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "imu_model.pth"

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print(f"   先训练: python3 train_imu.py 数据.csv")
        sys.exit(1)

    print(f"🧠 加载模型: {model_path}")
    print(f"📄 数据: {csv_path}")
    print()

    traj = predict_from_csv(csv_path, model_path)
    plot_trajectory(traj)

    # 输出统计
    total_dist = sum(math.sqrt((traj[i][0]-traj[i-1][0])**2 + (traj[i][1]-traj[i-1][1])**2)
                     for i in range(1, len(traj)))
    print(f"\n📊 统计:")
    print(f"   总位移: {math.sqrt(traj[-1][0]**2 + traj[-1][1]**2):.1f} 米")
    print(f"   总路程: {total_dist:.1f} 米")
    print(f"   终点: ({traj[-1][0]:.1f}, {traj[-1][1]:.1f})")
