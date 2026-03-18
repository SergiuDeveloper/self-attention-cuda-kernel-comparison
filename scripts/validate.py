import torch
import torch.nn.functional as F

from kernels.self_attention_torch import self_attention as attn_torch
from kernels.self_attention_triton import self_attention as attn_triton
from kernels.self_attention_cuda import self_attention as attn_cuda
from kernels.self_attention_numba import self_attention as attn_numba

ERR = 1e-6
BATCH_SIZE, SEQ_LEN, D_K = 16, 16, 64

if __name__ == '__main__':
    assert torch.cuda.is_available()

    q = torch.randn(BATCH_SIZE, SEQ_LEN, D_K).cuda()
    k = torch.randn(BATCH_SIZE, SEQ_LEN, D_K).cuda()
    v = torch.randn(BATCH_SIZE, SEQ_LEN, D_K).cuda()

    # Mask the first half of the token sequence
    mask = torch.zeros(SEQ_LEN, SEQ_LEN, dtype=torch.bool).cuda()
    mask[:, :SEQ_LEN // 2] = True
    mask = mask.unsqueeze(0).expand(BATCH_SIZE, -1, -1)
    additive_mask = torch.zeros(BATCH_SIZE, SEQ_LEN, SEQ_LEN, device='cuda').masked_fill(~mask, float('-inf'))

    ground_truth_result_no_mask = F.scaled_dot_product_attention(q, k, v)
    ground_truth_result_mask = F.scaled_dot_product_attention(q, k, v, attn_mask=additive_mask)

    assert torch.allclose(attn_torch(q, k, v), ground_truth_result_no_mask, atol=ERR)
    assert torch.allclose(attn_torch(q, k, v, mask), ground_truth_result_mask, atol=ERR)

    assert torch.allclose(attn_triton(q, k, v), ground_truth_result_no_mask, atol=ERR)
    assert torch.allclose(attn_triton(q, k, v, mask), ground_truth_result_mask, atol=ERR)

    assert torch.allclose(attn_cuda(q, k, v), ground_truth_result_no_mask, atol=ERR)
    assert torch.allclose(attn_cuda(q, k, v, mask), ground_truth_result_mask, atol=ERR)

    assert torch.allclose(attn_numba(q, k, v), ground_truth_result_no_mask, atol=ERR)
    assert torch.allclose(attn_numba(q, k, v, mask), ground_truth_result_mask, atol=ERR)
