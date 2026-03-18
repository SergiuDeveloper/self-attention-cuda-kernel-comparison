import math
import torch
import numba
import numpy as np
from numba import cuda

TILE_J = 8
MAX_D = 256
SMEM_SIZE = TILE_J * (MAX_D // 32 + 1)  # TILE_J * num_warps warp sums + TILE_J broadcast scores
NEG_INF = np.float32(np.finfo(np.float32).min)

@cuda.jit
def _attn_kernel(
    q, k, v, o,
    mask, has_mask,
    B, N, D, scale,
    sq0, sq1, sq2,
    sk0, sk1, sk2,
    sv0, sv1, sv2,
    so0, so1, so2,
    sm0, sm1, sm2
):
    # Shared memory
    smem = cuda.shared.array(SMEM_SIZE, dtype=numba.float32)

    d = cuda.threadIdx.x
    i = cuda.blockIdx.x
    b = cuda.blockIdx.y

    if b >= B or i >= N or d >= D:
        return

    # lane: position of this thread within its warp (0-31)
    # warp: which warp this thread belongs to
    # num_warps: how many warps exist in this block (one block = one output row)
    lane = d & 31
    warp = d >> 5
    num_warps = D >> 5

    q_d = q[b * sq0 + i * sq1 + d * sq2]

    m = NEG_INF
    l = np.float32(0.0)
    acc = np.float32(0.0)

    for j0 in range(0, N, TILE_J):
        # scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        # Each score is a dot product reduced across D threads via warp shuffles.
        # p0..p7 are written as separate variables instead of a loop because
        # Numba can't unroll a loop over mutable scalars in a GPU kernel -
        # each pN needs its own register for the shuffle chain to work correctly.
        p0 = q_d * (k[b*sk0 + (j0+0)*sk1 + d*sk2] if j0+0 < N else np.float32(0.0))
        p0 += cuda.shfl_xor_sync(0xffffffff, p0, 16)
        p0 += cuda.shfl_xor_sync(0xffffffff, p0, 8)
        p0 += cuda.shfl_xor_sync(0xffffffff, p0, 4)
        p0 += cuda.shfl_xor_sync(0xffffffff, p0, 2)
        p0 += cuda.shfl_xor_sync(0xffffffff, p0, 1)

        p1 = q_d * (k[b*sk0 + (j0+1)*sk1 + d*sk2] if j0+1 < N else np.float32(0.0))
        p1 += cuda.shfl_xor_sync(0xffffffff, p1, 16)
        p1 += cuda.shfl_xor_sync(0xffffffff, p1, 8)
        p1 += cuda.shfl_xor_sync(0xffffffff, p1, 4)
        p1 += cuda.shfl_xor_sync(0xffffffff, p1, 2)
        p1 += cuda.shfl_xor_sync(0xffffffff, p1, 1)

        p2 = q_d * (k[b*sk0 + (j0+2)*sk1 + d*sk2] if j0+2 < N else np.float32(0.0))
        p2 += cuda.shfl_xor_sync(0xffffffff, p2, 16)
        p2 += cuda.shfl_xor_sync(0xffffffff, p2, 8)
        p2 += cuda.shfl_xor_sync(0xffffffff, p2, 4)
        p2 += cuda.shfl_xor_sync(0xffffffff, p2, 2)
        p2 += cuda.shfl_xor_sync(0xffffffff, p2, 1)

        p3 = q_d * (k[b*sk0 + (j0+3)*sk1 + d*sk2] if j0+3 < N else np.float32(0.0))
        p3 += cuda.shfl_xor_sync(0xffffffff, p3, 16)
        p3 += cuda.shfl_xor_sync(0xffffffff, p3, 8)
        p3 += cuda.shfl_xor_sync(0xffffffff, p3, 4)
        p3 += cuda.shfl_xor_sync(0xffffffff, p3, 2)
        p3 += cuda.shfl_xor_sync(0xffffffff, p3, 1)

        p4 = q_d * (k[b*sk0 + (j0+4)*sk1 + d*sk2] if j0+4 < N else np.float32(0.0))
        p4 += cuda.shfl_xor_sync(0xffffffff, p4, 16)
        p4 += cuda.shfl_xor_sync(0xffffffff, p4, 8)
        p4 += cuda.shfl_xor_sync(0xffffffff, p4, 4)
        p4 += cuda.shfl_xor_sync(0xffffffff, p4, 2)
        p4 += cuda.shfl_xor_sync(0xffffffff, p4, 1)

        p5 = q_d * (k[b*sk0 + (j0+5)*sk1 + d*sk2] if j0+5 < N else np.float32(0.0))
        p5 += cuda.shfl_xor_sync(0xffffffff, p5, 16)
        p5 += cuda.shfl_xor_sync(0xffffffff, p5, 8)
        p5 += cuda.shfl_xor_sync(0xffffffff, p5, 4)
        p5 += cuda.shfl_xor_sync(0xffffffff, p5, 2)
        p5 += cuda.shfl_xor_sync(0xffffffff, p5, 1)

        p6 = q_d * (k[b*sk0 + (j0+6)*sk1 + d*sk2] if j0+6 < N else np.float32(0.0))
        p6 += cuda.shfl_xor_sync(0xffffffff, p6, 16)
        p6 += cuda.shfl_xor_sync(0xffffffff, p6, 8)
        p6 += cuda.shfl_xor_sync(0xffffffff, p6, 4)
        p6 += cuda.shfl_xor_sync(0xffffffff, p6, 2)
        p6 += cuda.shfl_xor_sync(0xffffffff, p6, 1)

        p7 = q_d * (k[b*sk0 + (j0+7)*sk1 + d*sk2] if j0+7 < N else np.float32(0.0))
        p7 += cuda.shfl_xor_sync(0xffffffff, p7, 16)
        p7 += cuda.shfl_xor_sync(0xffffffff, p7, 8)
        p7 += cuda.shfl_xor_sync(0xffffffff, p7, 4)
        p7 += cuda.shfl_xor_sync(0xffffffff, p7, 2)
        p7 += cuda.shfl_xor_sync(0xffffffff, p7, 1)

        if lane == 0:
            smem[0 * num_warps + warp] = p0
            smem[1 * num_warps + warp] = p1
            smem[2 * num_warps + warp] = p2
            smem[3 * num_warps + warp] = p3
            smem[4 * num_warps + warp] = p4
            smem[5 * num_warps + warp] = p5
            smem[6 * num_warps + warp] = p6
            smem[7 * num_warps + warp] = p7
        cuda.syncthreads()

        # scores = scores.masked_fill(mask == 0, float('-inf'))
        # weights = F.softmax(scores, dim=-1) - combine warp sums, broadcast
        if d == 0:
            for t in range(TILE_J):
                j = j0 + t
                s = np.float32(0.0)
                for w in range(num_warps):
                    s += smem[t * num_warps + w]
                s *= scale
                if j >= N or (has_mask and mask[b*sm0 + i*sm1 + j*sm2] == 0):
                    s = NEG_INF
                smem[TILE_J * num_warps + t] = s
        cuda.syncthreads()

        # weights @ v - online softmax update per tile element
        for t in range(TILE_J):
            j = j0 + t
            if j >= N:
                break
            score = smem[TILE_J * num_warps + t]
            m_new = max(m, score)
            rescale = math.exp(m - m_new)
            exp_s = math.exp(score - m_new)
            l = l * rescale + exp_s
            acc = acc * rescale + exp_s * v[b*sv0 + j*sv1 + d*sv2]
            m = m_new

    o[b * so0 + i * so1 + d * so2] = acc / l

def self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    B, N, D = q.shape
    o = torch.empty_like(q)
    scale = np.float32(1.0 / math.sqrt(D))

    qc = q.contiguous()
    kc = k.contiguous()
    vc = v.contiguous()

    has_mask = int(mask is not None)
    if mask is not None:
        mc = mask.contiguous()
        sm = (int(mc.stride(0)), int(mc.stride(1)), int(mc.stride(2)))
    else:
        mc = q.new_empty(1, dtype=torch.uint8)
        sm = (0, 0, 0)

    _attn_kernel[(N, B), D](
        cuda.as_cuda_array(qc.view(-1)),
        cuda.as_cuda_array(kc.view(-1)),
        cuda.as_cuda_array(vc.view(-1)),
        cuda.as_cuda_array(o.view(-1)),
        cuda.as_cuda_array(mc.view(-1)),
        has_mask, B, N, D, scale,
        int(qc.stride(0)), int(qc.stride(1)), int(qc.stride(2)),
        int(kc.stride(0)), int(kc.stride(1)), int(kc.stride(2)),
        int(vc.stride(0)), int(vc.stride(1)), int(vc.stride(2)),
        int(o.stride(0)), int(o.stride(1)), int(o.stride(2)),
        sm[0], sm[1], sm[2],
    )
    return o
