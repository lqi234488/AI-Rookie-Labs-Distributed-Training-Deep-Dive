import os, torch
import torch.distributed as dist
import torch.multiprocessing as mp

def fsdp_memory_mgmt(rank, ws):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("gloo", rank=rank, world_size=ws)

    # 1. 靜態存儲：僅保管專屬切片 (Static Shard)
    my_shard = torch.tensor([float(rank + 1)])
    print(f"[節點 {rank}] [靜態] 切片: {my_shard.tolist()}")

    # 2. All-Gather 收集全域參數 (Dynamic All-Gather)
    full_list = [torch.zeros(1) for _ in range(ws)]
    
    # TODO: 執行 All-Gather，把各節點的 my_shard 收集到 full_list
    dist.all_gather(full_list, my_shard)
    
    # TODO: 用 torch.cat 組合 full_weight，將 list 轉為一個連續的張量
    full_weight = torch.cat(full_list)
    
    print(f"[節點 {rank}] [配置] 完整: {full_weight.tolist()}")

    # --- 模擬 Forward/Backward ---
    # 在真實 FSDP 中，這部分會消耗大量 GPU VRAM
    print(f"[節點 {rank}] [運算] 完成")

    # 3. 記憶體釋放 (Memory Discarding)
    # TODO: 運算結束後立即刪除完整參數，釋放空間
    del full_weight
    
    try:
        # 測試是否還能讀取
        _ = full_weight
    except NameError:
        print(f"[節點 {rank}] [回收] 記憶體已釋放")

    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(fsdp_memory_mgmt, args=(2,), nprocs=2, join=True)