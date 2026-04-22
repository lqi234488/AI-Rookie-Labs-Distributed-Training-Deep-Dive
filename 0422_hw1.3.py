def all_gather_py(rank_data):
    # 這是作業 1.1 的實作：將碎片合併成完整清單
    gathered_result = [item for sublist in rank_data for item in sublist]
    return [gathered_result for _ in range(len(rank_data))]

def reduce_scatter_sum_py(rank_data):
    # 這是作業 1.2 的實作：逐元素加總並切分
    world_size = len(rank_data)
    data_length = len(rank_data[0])
    reduced_sum = [0] * data_length
    for data in rank_data:
        for i in range(data_length):
            reduced_sum[i] += data[i]
    return [[reduced_sum[i]] for i in range(world_size)]

def all_reduce_via_two_steps(rank_data):
    """
    透過兩個步驟實作 All-Reduce
    """
    # Step 1: 先 Reduce-Scatter
    # 每個 rank 會得到加總後的「一部分」：[[10], [10], [10], [10]]
    scattered_data = reduce_scatter_sum_py(rank_data)
    
    # Step 2: 再 All-Gather
    # 將每個人手上的 [10] 收集起來，重新拼成 [10, 10, 10, 10]
    final_result = all_gather_py(scattered_data)
    
    return final_result

# 測試
rank_data = [
    [1, 1, 1, 1], 
    [2, 2, 2, 2],
    [3, 3, 3, 3], 
    [4, 4, 4, 4]
]

result = all_reduce_via_two_steps(rank_data)

# 印出結果
for i, data in enumerate(result):
    print(f"rank{i}: {data}")