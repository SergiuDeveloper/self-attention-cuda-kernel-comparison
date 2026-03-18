import time
import torch
import numpy as np
import pandas as pd
from typing import Callable
from tqdm import tqdm

from kernels.self_attention_torch import self_attention as attn_torch
from kernels.self_attention_triton import self_attention as attn_triton
from kernels.self_attention_cuda import self_attention as attn_cuda
from kernels.self_attention_numba import self_attention as attn_numba

TIMING_RUNS = 20
WARMUP_RUNS = 5
COOLDOWN_SECS = 1.0

SEQ_LENS = [64, 128, 256]
BATCH_SIZES = [64, 128, 256]
D_KS = [64, 128, 256]

METHODS = {
    'Triton': attn_triton,
    'Numba': attn_numba,
    'CUDA': attn_cuda,
}

def run_timing(fn: Callable, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> np.ndarray:
    print('Warmup stage')
    for _ in tqdm(range(WARMUP_RUNS)):
        fn(q, k, v)
    torch.cuda.synchronize()

    print('Measurement stage')
    times_ms = []
    for _ in tqdm(range(TIMING_RUNS)):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)

        t0.record()
        fn(q, k, v)
        t1.record()

        torch.cuda.synchronize()
        times_ms.append(t0.elapsed_time(t1))

    return np.array(times_ms)

if __name__ == '__main__':
    assert torch.cuda.is_available()

    records = []
    for seq_len in SEQ_LENS:
        for batch_size in BATCH_SIZES:
            for d_k in D_KS:
                print(f'Sequence length {seq_len}, batch size {batch_size}, d_k {d_k}')
                q = torch.randn(batch_size, seq_len, d_k, device='cuda')
                k = torch.randn_like(q)
                v = torch.randn_like(q)

                for name, fn in METHODS.items():
                    print(f'Benchmarking kernel for {name}')
                    times = run_timing(fn, q, k, v)
                    print()
                    records.append({'seq_len': seq_len, 'batch_size': batch_size, 'd_k': d_k, 'method': name, 'mean_ms': float(np.mean(times))})
                    time.sleep(COOLDOWN_SECS)

                del q, k, v
                torch.cuda.empty_cache()

    df = pd.DataFrame(records).pivot(index=['seq_len', 'batch_size', 'd_k'], columns='method', values='mean_ms')
    df.columns.name = None
    print(df.to_string(float_format='{:.2f}'.format))
