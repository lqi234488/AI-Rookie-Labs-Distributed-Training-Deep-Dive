import torch
import torch.nn as nn

torch.manual_seed(42)

# 輸入 X (8 筆資料)
X = torch.randn(8, 6)
target = torch.randn(8, 2)

stage_1 = nn.Linear(6, 4)
stage_2 = nn.Linear(4, 2)

# --- 單卡標準答案 ---
Y_ref = stage_2(torch.relu(stage_1(X)))
loss_ref = nn.MSELoss()(Y_ref, target)
print(f"Single loss: {loss_ref.item():.4f}")

M = 4 # micro-batch 數量
batch_size = X.shape[0]
micro_size = batch_size // M

# 任務 1: 切成 M 個 micro-batch
# 使用 torch.chunk 按維度 0 切分，每份大小為 2
X_micros = torch.chunk(X, M, dim=0)
T_micros = torch.chunk(target, M, dim=0)

# 任務 2: 逐 micro-batch 過 pipeline
total_loss = torch.tensor(0.0)
for i in range(M):
    # 前向傳遞
    h = torch.relu(stage_1(X_micros[i]))
    out = stage_2(h)
    
    # 計算該 micro-batch 的 loss
    micro_loss = nn.MSELoss()(out, T_micros[i])
    
    # 任務 3: Loss 聚合
    # 因為 MSELoss 預設是取 Mean，我們需要除以 M 以確保總和與單卡一致
    total_loss = total_loss + (micro_loss / M)

print(f"GPipe loss: {total_loss.item():.4f}")

# 驗證
assert torch.allclose(loss_ref, total_loss)