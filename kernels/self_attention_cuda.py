import os
import torch
import ctypes
import subprocess

dir = os.path.dirname(os.path.abspath(__file__))
cu = os.path.join(dir, 'self_attention_cuda.cu')
dll = os.path.join(dir, 'self_attention_cuda.dll')

if not os.path.exists(dll) or os.path.getmtime(cu) > os.path.getmtime(dll):
    subprocess.check_call(['nvcc', '-O2', '--shared', '-o', dll, cu])

lib = ctypes.CDLL(dll, mode=ctypes.RTLD_GLOBAL)
lib.self_attention_forward.restype = None
lib.self_attention_forward.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, # Q K V O
    ctypes.c_void_p, ctypes.c_int, # Mask, has_mask
    ctypes.c_int, ctypes.c_int, ctypes.c_int, # B N D
    ctypes.c_int, ctypes.c_int, ctypes.c_int, # q strides
    ctypes.c_int, ctypes.c_int, ctypes.c_int, # k strides
    ctypes.c_int, ctypes.c_int, ctypes.c_int, # v strides
    ctypes.c_int, ctypes.c_int, ctypes.c_int, # o strides
    ctypes.c_int, ctypes.c_int, ctypes.c_int, # mask strides
]

def self_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    b, n, d = q.shape
    o = torch.empty_like(q)
    sm = (int(mask.stride(0)), int(mask.stride(1)), int(mask.stride(2))) if mask is not None else (0, 0, 0)

    lib.self_attention_forward(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(),
        mask.data_ptr() if mask is not None else q.data_ptr(),
        int(mask is not None),
        b, n, d,
        int(q.stride(0)), int(q.stride(1)), int(q.stride(2)),
        int(k.stride(0)), int(k.stride(1)), int(k.stride(2)),
        int(v.stride(0)), int(v.stride(1)), int(v.stride(2)),
        int(o.stride(0)), int(o.stride(1)), int(o.stride(2)),
        sm[0], sm[1], sm[2],
    )
    
    return o
