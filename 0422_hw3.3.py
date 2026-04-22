import torch
import torch.nn.functional as F

torch.manual_seed(42)

# 輸入 X (Batch=4, Hidden=8)
X = torch.randn(4, 8)
# 第一層權重 A (8, 12), 第二層權重 B (12, 6)
A = torch.randn(8, 12) 
B = torch.randn(12, 6)

# 標準答案 (單卡計算)
Y_ref = F.gelu(X @ A) @ B 

# Step 1: A 列切分 (將 8x12 切成兩個 8x6)
# 這樣每個裝置負責輸出 Hidden 維度的一半
A1, A2 = torch.chunk(A, 2, dim=1)

# Step 2: X @ Ai → GeLU (可獨立做)
# 關鍵：由於是列切分，GeLU 作用在每一列上，彼此不干擾
H1 = F.gelu(X @ A1)
H2 = F.gelu(X @ A2)

# Step 3: B 行切分 (將 12x6 切成兩個 6x6)
# 為了接住 H1 (4x6) 和 H2 (4x6)，B 必須按行切
B1, B2 = torch.chunk(B, 2, dim=0)

# Step 4: 各裝置計算部分和 (Partial Sum)
# 裝置 0 算 Z1，裝置 1 算 Z2
Z1 = H1 @ B1
Z2 = H2 @ B2

# Step 5: All-Reduce (Sum)
# 將所有裝置的部分和加總，得到最終結果
Y_tp = Z1 + Z2

# 驗證結果
assert torch.allclose(Y_ref, Y_tp, atol=1e-5)
print("✓ Megatron MLP TP 驗證通過！")