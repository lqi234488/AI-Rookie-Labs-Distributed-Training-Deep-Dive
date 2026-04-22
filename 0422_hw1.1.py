def all_gather_py(rank_data):
    """
    每個 rank 都拿到所有人的資料。
    rank_data: list of lists，例如 [[1], [2], [3], [4]]
    """
    # 1. 將所有 rank 的資料合併成一個完整的清單
    # 使用列表推導式將嵌套列表展平
    gathered_result = [item for sublist in rank_data for item in sublist]
    
    # 2. 模擬分散式環境中，每個 rank 都拿到了這份資料
    # 在真實環境中，這通常涉及網路通訊 (MPI_Allgather)
    final_output = []
    for _ in range(len(rank_data)):
        final_output.append(gathered_result)
        
    return final_output

# 測試
rank_data = [[1], [2], [3], [4]]
result = all_gather_py(rank_data)

# 印出結果
for i, data in enumerate(result):
    print(f"rank{i}: {data}")