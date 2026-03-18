import math
import torch
import triton
import triton.language as tl

@triton.jit
def _attn_kernel(
    q, k, v, o, mask,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    stride_mb, stride_mn, stride_mk,
    N, D, scale,
    has_mask: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr
):
    b = tl.program_id(0)
    i = tl.program_id(1)

    offs_d = tl.arange(0, block_d)
    d_mask = offs_d < D

    q_row = tl.load(
        q + b * stride_qb + i * stride_qn + offs_d * stride_qd,
        mask=d_mask, other=0.0
    )

    m = float('-inf')
    l = 0.0
    acc = tl.zeros([block_d], dtype=tl.float32)

    for j in range(0, N, block_n):
        offs_n = j + tl.arange(0, block_n)
        n_mask = offs_n < N

        k_tile = tl.load(
            k + b * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0
        )
        # scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        scores = tl.sum(q_row[None, :] * k_tile, axis=1) * scale
        scores = tl.where(n_mask, scores, float('-inf'))

        if has_mask:
            m_row = tl.load(
                mask + b * stride_mb + i * stride_mn + offs_n * stride_mk,
                mask=n_mask, other=False
            )
            # scores = scores.masked_fill(mask == 0, float('-inf'))
            scores = tl.where(m_row, scores, float('-inf'))

        # weights = F.softmax(scores, dim=-1) - online softmax update
        m_new = tl.maximum(m, tl.max(scores, axis=0))
        rescale = tl.exp(m - m_new)
        exp_scores = tl.where(n_mask, tl.exp(scores - m_new), 0.0)

        v_tile = tl.load(
            v + b * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None] & d_mask[None, :], other=0.0
        )
        # weights @ v
        acc = acc * rescale + tl.sum(exp_scores[:, None] * v_tile, axis=0)
        l = l * rescale + tl.sum(exp_scores, axis=0)
        m = m_new

    tl.store(
        o + b * stride_ob + i * stride_on + offs_d * stride_od,
        acc / l,
        mask=d_mask
    )

def self_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None
) -> torch.Tensor:
    b, n, d = q.shape
    o = torch.empty_like(q)

    block_n = min(64, triton.next_power_of_2(n))
    block_d = triton.next_power_of_2(d)

    if mask is not None:
        _attn_kernel[(b, n)](
            q, k, v, o, mask,
            *q.stride(), *k.stride(), *v.stride(), *o.stride(), *mask.stride(),
            n, d, 1.0 / math.sqrt(d),
            has_mask=True, block_n=block_n, block_d=block_d
        )
    else:
        dummy = q.new_empty(0, dtype=torch.bool)
        _attn_kernel[(b, n)](
            q, k, v, o, dummy,
            *q.stride(), *k.stride(), *v.stride(), *o.stride(), 0, 0, 0,
            n, d, 1.0 / math.sqrt(d),
            has_mask=False, block_n=block_n, block_d=block_d
        )

    return o
