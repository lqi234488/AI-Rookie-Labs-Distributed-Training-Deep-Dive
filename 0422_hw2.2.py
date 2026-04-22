import os, torch
import torch.distributed as dist
import torch.multiprocessing as mp

def ddp_sync_simulation(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    
    # 初始化處理群組
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    my_weight = torch.tensor([10.0])
    # Rank 0 梯度是 2.0，Rank 1 梯度是 8.0
    my_gradient = torch.tensor([2.0 if rank == 0 else 8.0])
    learning_rate = 0.1

    # 任務 1: All-Reduce 全域梯度同步
    # 將所有節點的 my_gradient 相加，結果會直接覆蓋原有的 my_gradient
    dist.all_reduce(my_gradient, op=dist.ReduceOp.SUM)
    # 此時所有節點的 my_gradient 都變成了 [10.0]

    # 任務 2: 計算全域平均梯度
    # 全域總和除以 world_size (2) 得到全域平均 [5.0]
    my_gradient = my_gradient / world_size

    # 任務 3: 各節點獨立更新權重
    # w = w - η * g_avg
    my_weight = my_weight - (learning_rate * my_gradient)

    print(f"[節點 {rank}] 權重: {my_weight.item()}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    # 啟動 2 個進程模擬分散式環境
    mp.spawn(ddp_sync_simulation, args=(2,), nprocs=2, join=True)