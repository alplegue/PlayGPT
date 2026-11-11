import time
import torch

''' Lets estimate the max FLOPS we can get on MPS. We will do a simple matrix multiplication of size N x N, 
and repeat it ITERS times. We will measure the time taken and compute the FLOPS as 2 * N^3 * ITERS / time_taken. 
The factor of 2 comes from the fact that each value of the resulting multiplication involves N multiplications and N-1 additions. 
We will do this for both float32 and bfloat16 data types. '''

N, ITERS = 4096, 50

for dtype in [torch.float32, torch.bfloat16]:
    a = torch.randn(N, N, device="mps", dtype=dtype)
    b = torch.randn(N, N, device="mps", dtype=dtype)
    for _ in range(5):  # warm-up
        a @ b
    torch.mps.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        a @ b
    torch.mps.synchronize()
    print(f"{dtype}: {2 * N**3 * ITERS / (time.time() - t0) / 1e12:.2f} TFLOP/s")

