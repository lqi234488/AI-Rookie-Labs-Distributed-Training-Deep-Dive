import torch
import torch.nn as nn

torch.manual_seed(42)

# 輸入 X 與目標 target
X = torch.randn(4, 8)
target = torch.randn(4, 2)

# 模型共有 4 層 Linear
layers = [
    nn.Linear(8, 6), 
    nn.Linear(6, 6),
    nn.Linear(6, 4), 
    nn.Linear(4, 2)
]

# 單卡標準答案 (Sequential 執行)
with torch.no_grad():
    Y_ref = X
    for layer in layers:
        Y_ref = torch.relu(layer(Y_ref))

# 任務 1: 每層分配給一個 Stage (模擬 4 個 GPU)
stage_0 = layers[0]
stage_1 = layers[1]
stage_2 = layers[2]
stage_3 = layers[3]

# 任務 2: 逐 Stage 前向傳遞 (模擬跨裝置通訊)
# 每個 Stage 接收前一個 Stage 的輸出，運算後再傳給下一個
act_0 = torch.relu(stage_0(X))     # Stage 0 運算
act_1 = torch.relu(stage_1(act_0)) # Stage 0 傳給 Stage 1
act_2 = torch.relu(stage_2(act_1)) # Stage 1 傳給 Stage 2
Y_pp  = torch.relu(stage_3(act_2)) # Stage 2 傳給 Stage 3

# 驗證
assert torch.allclose(Y_ref, Y_pp)
print("✓ Pipeline 前向傳遞驗證通過！")

# 額外資訊：參數量印出 (Weight + Bias)
print(f"Stage 0 參數量: {sum(p.numel() for p in stage_0.parameters())}")
print(f"Stage 3 參數量: {sum(p.numel() for p in stage_3.parameters())}")