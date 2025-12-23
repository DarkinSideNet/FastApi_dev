import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File
from sklearn.preprocessing import StandardScaler
import io

app = FastAPI(title="Weather Prediction TCN API")

# --- 1. Kiến trúc Model (Phải khớp với lúc train) ---
class TCN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_inputs, 32, 3, padding=2, dilation=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=4, dilation=2),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64, num_outputs)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.net(x)
        return self.fc(y[:, :, -1])

# --- 2. Load Model & Scaler ---
MODEL_PATH = "model.pth"
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)

# Khởi tạo model và nạp trọng số
features = checkpoint["features"]
targets = checkpoint["targets"]
model = TCN(len(features), len(targets))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Tái tạo Scaler từ thông số đã lưu
scaler_X = StandardScaler()
scaler_X.mean_ = np.array(checkpoint["scaler_mean"])
scaler_X.scale_ = np.array(checkpoint["scaler_scale"])
scaler_X.n_features_in_ = len(features)

seq_len = checkpoint["config"]["seq_len"]

# --- 3. API Endpoints ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Đọc file CSV từ người dùng
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Kiểm tra đủ số dòng tối thiểu (bằng seq_len)
    if len(df) < seq_len:
        return {"error": f"Cần tối thiểu {seq_len} dòng dữ liệu để dự báo."}

    # Lấy dữ liệu cuối cùng để dự báo
    data_tail = df[features].tail(seq_len).values
    X_scaled = scaler_X.transform(data_tail)
    X_tensor = torch.tensor([X_scaled], dtype=torch.float32) # Shape: [1, seq_len, n_features]

    # Chạy Inference
    with torch.no_grad():
        preds = model(X_tensor).numpy()[0]

    # Map kết quả với tên target
    result = {targets[i]: float(preds[i]) for i in range(len(targets))}
    return {"prediction": result, "model_used": MODEL_PATH}

@app.get("/")
def health_check():
    return {"status": "online"}