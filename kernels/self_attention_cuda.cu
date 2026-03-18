#include <cuda_runtime.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

#define TILE_J 8

// Inline method for performance
#define WARP_REDUCE_SUM(val)                         \
    (val) += __shfl_xor_sync(0xffffffff, (val), 16); \
    (val) += __shfl_xor_sync(0xffffffff, (val), 8);  \
    (val) += __shfl_xor_sync(0xffffffff, (val), 4);  \
    (val) += __shfl_xor_sync(0xffffffff, (val), 2);  \
    (val) += __shfl_xor_sync(0xffffffff, (val), 1);

// One block per (b, i) output row, D threads (one per output dim).
// Computing each attention score requires a dot product across all D threads,
// which we reduce with warp shuffles (no barrier) + 2 __syncthreads to collect
// and broadcast the result. We process TILE_J keys per loop iteration to
// amortize those 2 barriers across TILE_J scores instead of paying them per key.
__global__ void _attn_kernel(
    const float *__restrict__ q,
    const float *__restrict__ k,
    const float *__restrict__ v,
    float *__restrict__ o,
    const uint8_t *mask,
    int N, int D, float scale, int has_mask,
    int sq0, int sq1, int sq2,
    int sk0, int sk1, int sk2,
    int sv0, int sv1, int sv2,
    int so0, int so1, int so2,
    int sm0, int sm1, int sm2)
{
    // Shared memory
    extern __shared__ float smem[];

    int b = blockIdx.y;
    int i = blockIdx.x;
    int d = threadIdx.x;
    int lane = d & 31;
    int warp = d >> 5;
    int num_warps = blockDim.x >> 5;

    // Load q[b,i,d] once into a register for the entire kernel.
    float q_d = __ldg(&q[b * sq0 + i * sq1 + d * sq2]);

    float m = -FLT_MAX;
    float l = 0.f;
    float out = 0.f;

    for (int j0 = 0; j0 < N; j0 += TILE_J)
    {
        // scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        // TILE_J warp reductions, no sync needed
        // partials[t] = warp-local partial sum of dot(q[b,i], k[b,j0+t]).
        // After WARP_REDUCE_SUM each lane 0 holds the full warp sum; other lanes
        // hold intermediate values that are discarded (only lane 0 writes to smem).
        float partials[TILE_J];
        for (int t = 0; t < TILE_J; ++t)
        {
            int j = j0 + t;
            float k_d = (j < N) ? __ldg(&k[b * sk0 + j * sk1 + d * sk2]) : 0.f;
            float p = q_d * k_d;
            WARP_REDUCE_SUM(p);
            partials[t] = p;
        }

        if (lane == 0)
            for (int t = 0; t < TILE_J; ++t)
                smem[t * num_warps + warp] = partials[t];
        __syncthreads();

        // scores = scores.masked_fill(mask == 0, float('-inf'))
        // weights = F.softmax(scores, dim=-1) - combine warp sums, broadcast
        if (d == 0)
            for (int t = 0; t < TILE_J; ++t)
            {
                int j = j0 + t;
                float s = 0.f;
                for (int w = 0; w < num_warps; ++w)
                    s += smem[t * num_warps + w];
                s *= scale;
                if (j >= N || (has_mask && !mask[b * sm0 + i * sm1 + j * sm2]))
                    s = -FLT_MAX;
                smem[TILE_J * num_warps + t] = s;
            }
        __syncthreads();

        // weights @ v - online softmax update per tile element
        for (int t = 0; t < TILE_J; ++t)
        {
            int j = j0 + t;
            if (j >= N)
                break;
            float score = smem[TILE_J * num_warps + t];
            float m_new = fmaxf(m, score);
            float rescale = expf(m - m_new);
            float exp_s = expf(score - m_new);
            l = l * rescale + exp_s;

            // Rescale the running output sum to the new max, then add the
            // contribution of key j: its softmax weight (exp_s) times v[b,j,d].
            // __ldg again uses the read-only cache since v is never written here.
            out = out * rescale + exp_s * __ldg(&v[b * sv0 + j * sv1 + d * sv2]);
            m = m_new;
        }
    }

    o[b * so0 + i * so1 + d * so2] = out / l;
}

extern "C"
{
    __declspec(dllexport) void self_attention_forward(
        float *q, float *k, float *v, float *o,
        uint8_t *mask, int has_mask,
        int B, int N, int D,
        int sq0, int sq1, int sq2,
        int sk0, int sk1, int sk2,
        int sv0, int sv1, int sv2,
        int so0, int so1, int so2,
        int sm0, int sm1, int sm2)
    {
        float scale = 1.f / sqrtf((float)D);
        dim3 block(D);
        dim3 grid(N, B);
        size_t smem = (TILE_J * (D / 32) + TILE_J) * sizeof(float);
        _attn_kernel<<<grid, block, smem>>>(
            q, k, v, o, mask, N, D, scale, has_mask,
            sq0, sq1, sq2, sk0, sk1, sk2,
            sv0, sv1, sv2, so0, so1, so2,
            sm0, sm1, sm2);
        cudaDeviceSynchronize();
    }
}
