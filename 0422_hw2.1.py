import torch

# 系統初始設定
main_weight = torch.tensor([10.0]) # 初始權重
learning_rate = 0.1

# 模擬 4 個 Worker 計算出的局部梯度
worker_gradients = [
    torch.tensor([2.0]), torch.tensor([4.0]),
    torch.tensor([6.0]), torch.tensor([8.0])
]

# 任務 1: 計算平均梯度 (Gradient Aggregation)
# 將所有 tensor 放入一個 list 後使用 torch.stack 並計算 mean
# 或者直接使用 sum() 除以長度
avg_grad = torch.stack(worker_gradients).mean()

# 任務 2: 執行權重更新 (Weight Update)
# 遵循 SGD 公式：w = w - η * g
main_weight = main_weight - (learning_rate * avg_grad)

print(f"[主節點] 更新完成，權重: {main_weight.item()}")