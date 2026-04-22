import torch

torch.manual_seed(42)

# 輸入矩陣 X: (4, 8)
X = torch.randn(4, 8)
# 權重矩陣 A: (8, 6)
A = torch.randn(8, 6)

# 標準答案
Y_ref = X @ A

# 任務 1: A 按行 (Row) 切分 (從 8x6 變成兩個 4x6)
# dim=0 代表按行切
A1, A2 = torch.chunk(A, 2, dim=0)

# 任務 2: X 按列 (Column) 切分 (從 4x8 變成兩個 4x4)
# 為了與 A 的行數對齊，X 必須切列 (dim=1)
X1, X2 = torch.chunk(X, 2, dim=1)

# 任務 3: 各「裝置」計算部分和 (Partial Sum)
# 裝置 0 計算 X1 @ A1 (4x4 @ 4x6 -> 4x6)
# 裝置 1 計算 X2 @ A2 (4x4 @ 4x6 -> 4x6)
Z1 = X1 @ A1
Z2 = X2 @ A2

# 任務 4: All-Reduce (Sum) 還原
# 將各裝置算出的部分結果相加
Y_tp = Z1 + Z2

# 驗證
assert torch.allclose(Y_ref, Y_tp)
print("✓ 行切分驗證通過！")