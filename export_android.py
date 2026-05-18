#!/usr/bin/env python3
"""导出 PyTorch 模型 → TorchScript（手机可用格式）"""
import torch, sys, os

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

def export(model_path="imu_model.pth", output_path="imu_model.pt"):
    device = torch.device("cpu")
    model = IMUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 用随机输入做 trace（窗口400帧）
    example = torch.rand(1, 400, 6)  # batch=1, window=400, channels=6
    traced = torch.jit.trace(model, example)

    traced.save(output_path)
    file_size = os.path.getsize(output_path)
    print(f"✅ 模型已导出: {output_path} ({file_size/1024:.1f}KB)")
    return output_path

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "imu_model.pth"
    export(path)
