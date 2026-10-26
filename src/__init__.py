"""Package init — runs before any experiment module, so these settings apply everywhere.

Colab ships JAX and TensorFlow with GPU support. JAX reserves 75% of the GPU the moment
it is first used (bm25s triggers this on import), which leaves PyTorch without memory.
We only use JAX for a tiny top-k sort, so it is pinned to the CPU.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def warn_if_gpu_busy(min_free_fraction: float = 0.8):
    """Print a clear warning if another process is already holding GPU memory."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("[gpu] WARNING: no GPU found — this will be very slow. Turn on the T4 GPU.")
            return
        free, total = torch.cuda.mem_get_info()
        print(f"[gpu] {torch.cuda.get_device_name(0)}: {free / 2**30:.1f} of {total / 2**30:.1f} GiB free")
        if free < min_free_fraction * total:
            print("[gpu] WARNING: another process is using the GPU. In Colab: Runtime -> Restart session, "
                  "then run the notebook again from Step 1.")
    except Exception as e:  # never block a run because of the check itself
        print(f"[gpu] check skipped ({e})")
