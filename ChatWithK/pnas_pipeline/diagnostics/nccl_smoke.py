"""Multi-GPU NCCL smoke test (WP0). Launch with torchrun, one process per GPU:

    torchrun --standalone --nproc_per_node=4 -m pnas_pipeline.diagnostics.nccl_smoke
"""
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    # Short timeout: a P2P hang should fail in a minute, not the 10-minute default.
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    x = torch.full((1,), float(rank + 1), device="cuda")
    dist.all_reduce(x)
    expected = world * (world + 1) / 2
    assert x.item() == expected, f"all_reduce gave {x.item()}, expected {expected}"
    payload = torch.ones(64 * 1024 * 1024, device="cuda")  # 256 MB fp32
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        dist.all_reduce(payload)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - t0) / 5
    # Ring all-reduce bus bandwidth, as reported by nccl-tests.
    busbw = payload.numel() * 4 * 2 * (world - 1) / world / seconds / 1e9
    if rank == 0:
        print(f"NCCL OK p2p_disable={os.environ.get('NCCL_P2P_DISABLE', '0')} world={world} all_reduce_256MB={seconds * 1e3:.1f} ms busbw={busbw:.1f} GB/s")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
