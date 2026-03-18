# self-attention-cuda-kernel-comparison

How fast can you implement self-attention from scratch in a few days, depending on the tool you use?

This repo benchmarks four implementations against each other: a hand-written CUDA C kernel, a Numba CUDA kernel, and a Triton kernel. All are validated against `torch.nn.functional.scaled_dot_product_attention` as ground truth.

## Implementations

| File                               | Description                                             |
| ---------------------------------- | ------------------------------------------------------- |
| `kernels/self_attention_cuda.cu`   | CUDA C kernel with warp shuffle reductions and j-tiling |
| `kernels/self_attention_numba.py`  | Numba CUDA kernel, same algorithm                       |
| `kernels/self_attention_triton.py` | Triton kernel with online softmax tiling                |
| `kernels/self_attention_torch.py`  | PyTorch reference (`q @ k.T` + softmax)                 |

## Results

Benchmarked on a 6 GB VRAM GPU. Numbers are mean latency in ms - lower is better.

```
                         CUDA   Numba  Triton
seq_len batch_size d_k
64      64         64    0.76    1.53    0.55
                   128   1.39    6.45    1.10
                   256   2.29    4.09    6.98
        128        64    2.38    2.13    8.12
                   128   2.38    4.10   10.42
                   256   3.79    5.70    6.40
        256        64    3.26    5.18    2.17
                   128   3.87    5.93    6.46
                   256   7.66   10.89    8.18
128     64         64    2.54    5.55    1.98
                   128   3.54    5.97    5.03
                   256   7.63   10.48    8.02
        128        64    3.98    6.97    4.08
                   128   7.13    9.94    7.98
                   256  15.15   20.38   15.79
        256        64    7.55   11.42    8.30
                   128  14.09   19.30   15.69
                   256  30.05   40.58   31.38
256     64         64    7.04    9.93    7.83
                   128  14.11   18.80   15.48
                   256  29.65   39.65   31.02
        128        64   14.03   19.27   15.65
                   128  30.15   39.80   30.97
                   256  73.56   94.00   61.95
        256        64   33.24   45.14   31.11
                   128  69.05   93.30   61.76
                   256 147.50  204.27  123.77
```

CUDA wins at small to mid sizes. Triton pulls ahead at large seq_len and d_k, where its tiled memory access pattern becomes more advantageous. Numba consistently trails both due to higher kernel launch and JIT overhead.

## Usage

Run correctness tests:

```bash
python -m scripts.test.py
```

Run the benchmark:

```bash
python -m scripts.benchmark.py
```
