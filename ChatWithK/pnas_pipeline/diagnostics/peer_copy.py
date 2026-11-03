"""Check direct GPU-to-GPU tensor copies (used by multi-GPU device_map models).

    python -m pnas_pipeline.diagnostics.peer_copy   # needs >= 2 visible GPUs
"""
import torch


def main():
    n = torch.cuda.device_count()
    assert n >= 2, "needs at least 2 GPUs"
    for src in range(n):
        for dst in range(n):
            if src == dst:
                continue
            x = torch.arange(32 * 1024 * 1024, device=src, dtype=torch.float32)
            y = x.to(dst)
            torch.cuda.synchronize(dst)
            ok = torch.equal(y.cpu(), x.cpu())
            print(f"{src}->{dst} can_access_peer={torch.cuda.can_device_access_peer(src, dst)} copy_ok={ok}",
                  flush=True)
            assert ok, f"corrupted copy {src}->{dst}"
    print("peer copy OK")


if __name__ == "__main__":
    main()
