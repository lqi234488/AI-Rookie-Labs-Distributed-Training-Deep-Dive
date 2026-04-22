import torch

torch.manual_seed(42)

# 輸入矩陣 X: (Batch_size=4, Hidden_dim=8)
X = torch.randn(4, 8) 
# 權重矩陣 A: (In_dim=8, Out_dim=6)
A = torch.randn(8, 6) 

# 標準答案 (完整運算)
Y_ref = X @ A 

# 任務 1: 將 A 按列切分成兩片 (每片從 8x6 變成 8x3)
# 使用 torch.chunk(tensor, chunks, dim)
A1, A2 = torch.chunk(A, 2, dim=1)

# 任務 2: 各「裝置」用完整 X 乘分片
# 裝置 0 計算 Y 的左半邊 (4x3)，裝置 1 計算 Y 的右半邊 (4x3)
Y1 = X @ A1
Y2 = X @ A2

# 任務 3: All-Gather — 按列 (dim=1) 拼接還原
# 在分散式環境中，這一步通常由 dist.all_gather 觸發，這裡用 torch.cat 模擬
Y_tp = torch.cat([Y1, Y2], dim=1)

# 驗證結果是否一致
assert torch.allclose(Y_ref, Y_tp)
print("✓ 列切分驗證通過！")