def reduce_scatter_sum_py(rank_data):
    """
    先逐元素 SUM，再切開每人只拿一份。
    rank_data: list of lists，例如 [[1,1,1,1], [2,2,2,2], [3,3,3,3], [4,4,4,4]]
    """
    world_size = len(rank_data)
    data_length = len(rank_data[0])
    
    # 1. Reduce: 逐元素相加 (Element-wise Sum)
    # 我們初始化一個長度相同的全 0 列表
    reduced_sum = [0] * data_length
    for data in rank_data:
        for i in range(data_length):
            reduced_sum[i] += data[i]
            
    # 此時 reduced_sum 為 [10, 10, 10, 10]
    
    # 2. Scatter: 每人拿走一份
    # 根據定義，rank i 會拿到 reduced_sum[i]
    # 因為輸出要求是 [10]，所以我們把每個元素包成 list
    final_output = []
    for i in range(world_size):
        final_output.append([reduced_sum[i]])
        
    return final_output

# 測試
rank_data = [
    [1, 1, 1, 1], 
    [2, 2, 2, 2],
    [3, 3, 3, 3], 
    [4, 4, 4, 4]
]
result = reduce_scatter_sum_py(rank_data)

# 印出結果
for i, val in enumerate(result):
    print(f"rank{i}: {val}")