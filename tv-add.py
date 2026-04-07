import torch
from functools import partial
from typing import List

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# Basic Kernel Implementation
# ---------------------
# This is our first implementation of the elementwise add kernel.
# It follows a simple 1:1 mapping between threads and tensor elements.


@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,  # Input tensor A
    gB: cute.Tensor,  # Input tensor B
    gC: cute.Tensor,  # Output tensor C = A + B
    tv_layout: cute.Layout # (thread_id, value_id) -> 2D coordinate in (TileM, TileN)
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = (None, None), bidx

    # the specific slice handled by one thread block
    blkA = gA[blk_coord] # 2D coordinate in (TileM, TileN) -> physical address
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]

    # compose for thread-index & value-index to physical mapping 
    # different physical address per block
    # frg_ : (thread_id, value_id) -> physical address
    # same offset, different base per block
    frgA = cute.composition(blkA, tv_layout)
    frgB = cute.composition(blkB, tv_layout)
    frgC = cute.composition(blkC, tv_layout)

    print(f"Composed with TV layout:")
    print(f"  frgA: {frgA.type}")

    # one thread, all values for this thread
    thr_coord = (tidx, None)

    thrA = frgA[thr_coord]
    thrB = frgB[thr_coord]
    thrC = frgC[thr_coord]
    
    thrC[None] = thrA.load() + thrB.load() # equivalent to thrC.store(thrA.load() + thrB.load()) 

@cute.jit
def elementwise_add(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    # mA layout: (M, N):(N, 1)
    # TV layout map thread & value index to (16, 256) logical tile
    #  - contiguous thread index maps to mode-1 because input layout is contiguous on
    #     mode-1 for coalesced load-store
    #  - each thread load 8 contiguous element each row and load 4 rows

    thr_layout = cute.make_layout(shape=(4, 32), stride=(32, 1))
    val_layout = cute.make_layout(shape=(4, 8), stride=(8, 1))

    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    print(f"Tiler: {tiler_mn}") # the tile shape for a block, (16, 256)
    print(f"TV Layout: {tv_layout}") # (thread_id, value_id) → (TileM, TileN) coordinates

    gA = cute.zipped_divide(mA, tiler_mn)
    gB = cute.zipped_divide(mB, tiler_mn)
    gC = cute.zipped_divide(mC, tiler_mn)

    print("[DSL INFO] Tiled Tensors:")
    print(f"[DSL INFO]   gA = {gA}")
    print(f"[DSL INFO]   gB = {gB}")
    print(f"[DSL INFO]   gC = {gC}")

    elementwise_add_kernel(gA, gB, gC, tv_layout).launch(
        grid=(cute.size(gC, mode=[1]), 1, 1),
        block=(cute.size(tv_layout, mode=[0]), 1, 1),
    )


M, N = 16384, 8192

a = torch.randn(M, N, device="cuda", dtype=torch.float16)
b = torch.randn(M, N, device="cuda", dtype=torch.float16)
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

a_ = from_dlpack(a, assumed_align=16)
b_ = from_dlpack(b, assumed_align=16)
c_ = from_dlpack(c, assumed_align=16)

compiled_func = cute.compile(elementwise_add, a_, b_, c_)
compiled_func(a_, b_, c_)

# verify correctness
torch.testing.assert_close(c, a + b)




# helper in benchmark
num_elements = sum([a.numel(), b.numel(), c.numel()])

def benchmark(callable, a_, b_, c_):
    avg_time_us = cute.testing.benchmark(
        callable,
        kernel_arguments=cute.testing.JitArguments(a_, b_, c_),
        warmup_iterations=5,
        iterations=100,
    )

    # Calculate metrics
    # ----------------
    dtype = a_.element_type

    # Calculate total bytes transferred:
    # - 2 reads (A and B) + 1 write (C)
    # - Each element is dtype.width bits
    bytes_per_element = dtype.width // 8
    total_bytes = num_elements * bytes_per_element

    # Calculate achieved bandwidth
    achieved_bandwidth = total_bytes / (avg_time_us * 1000)  # GB/s

    # Print results
    # ------------
    print(f"Performance Metrics:")
    print(f"-------------------")
    print(f"Kernel execution time: {avg_time_us:.4f} us")
    print(f"Memory throughput: {achieved_bandwidth:.2f} GB/s")

benchmark(compiled_func, a_, b_, c_)

