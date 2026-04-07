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
def naive_elementwise_add_kernel(
    gA: cute.Tensor,  # Input tensor A
    gB: cute.Tensor,  # Input tensor B
    gC: cute.Tensor,  # Output tensor C = A + B
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    m, n = gA.shape # M rows, N columns

    global_idx = bdimx * bidx + tidx

    mi = global_idx // n
    ni = global_idx % n

    a = gA[mi, ni]
    b = gB[mi, ni]
    gC[mi, ni] = a + b


@cute.jit  # Just-in-time compilation decorator
def naive_elementwise_add(
    mA: cute.Tensor,  # Input tensor A
    mB: cute.Tensor,  # Input tensor B
    mC: cute.Tensor,  # Output tensor C
):
    num_threads_per_block = 256
    print(f"[DSL INFO] Input tensors:")
    print(f"[DSL INFO]   mA = {mA}")
    print(f"[DSL INFO]   mB = {mB}")
    print(f"[DSL INFO]   mC = {mC}")
    m, n = mA.shape  # Matrix dimensions (M rows × N columns)
    kernel = naive_elementwise_add_kernel(mA, mB, mC)
    kernel.launch(
        grid=((m * n) // num_threads_per_block, 1, 1),  # Number of blocks in x,y,z
        block=(num_threads_per_block, 1, 1),  # Threads per block in x,y,z
    )

# Test Setup
# ----------
# Define test dimensions
M, N = 16384, 8192  # Using large matrices to measure performance

# Create test data on GPU
# ----------------------
# Using float16 (half precision) for:
# - Reduced memory bandwidth requirements
# - Better performance on modern GPUs
a = torch.randn(M, N, device="cuda", dtype=torch.float16)  # Random input A
b = torch.randn(M, N, device="cuda", dtype=torch.float16)  # Random input B
c = torch.zeros(M, N, device="cuda", dtype=torch.float16)  # Output buffer

# Calculate total elements for bandwidth calculations
num_elements = sum([a.numel(), b.numel(), c.numel()])

# Convert PyTorch tensors to CuTe tensors
# -------------------------------------
# from_dlpack creates CuTe tensor views of PyTorch tensors
# assumed_align=16 ensures proper memory alignment for vectorized access
a_ = from_dlpack(a, assumed_align=16)  # CuTe tensor A
b_ = from_dlpack(b, assumed_align=16)  # CuTe tensor B
c_ = from_dlpack(c, assumed_align=16)  # CuTe tensor C

# Compile the kernel for the specific input types
naive_elementwise_add_ = cute.compile(naive_elementwise_add, a_, b_, c_)

# Run the kernel
naive_elementwise_add_(a_, b_, c_)

# Verify Results
# -------------
# Compare our kernel output with PyTorch's native implementation
torch.testing.assert_close(c, a + b)  # Raises error if results don't match

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

benchmark(naive_elementwise_add_, a_, b_, c_)
