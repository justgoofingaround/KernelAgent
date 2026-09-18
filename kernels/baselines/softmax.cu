// Row-wise softmax, one thread block per row.
//
// Uses the "online softmax" trick (Milakov & Gimelshein, 2018): each thread
// tracks a running (max, sum) pair in a single pass over the row, so the input
// is read twice (reduce + write) instead of three times (max, sum, write).
// All math is done in fp32 regardless of the storage dtype.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

namespace {

constexpr unsigned kFullMask = 0xffffffffu;

// Running softmax state: m = max seen so far, d = sum of exp(x - m).
struct MaxSum {
  float m;
  float d;
};

__device__ __forceinline__ MaxSum combine(MaxSum a, MaxSum b) {
  const float m = fmaxf(a.m, b.m);
  if (m == -INFINITY) return {m, 0.f};  // both empty: avoid exp(-inf - -inf) = NaN
  return {m, a.d * expf(a.m - m) + b.d * expf(b.m - m)};
}

__device__ __forceinline__ MaxSum warp_reduce(MaxSum v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    MaxSum other{__shfl_xor_sync(kFullMask, v.m, offset), __shfl_xor_sync(kFullMask, v.d, offset)};
    v = combine(v, other);
  }
  return v;
}

// Reduce across the block: warps reduce with shuffles, then warp 0 reduces the
// per-warp partials from shared memory. Every thread gets the final value.
__device__ __forceinline__ MaxSum block_reduce(MaxSum v) {
  __shared__ MaxSum partials[32];
  __shared__ MaxSum result;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;

  v = warp_reduce(v);
  if (lane == 0) partials[warp] = v;
  __syncthreads();

  if (warp == 0) {
    v = lane < num_warps ? partials[lane] : MaxSum{-INFINITY, 0.f};
    v = warp_reduce(v);
    if (lane == 0) result = v;
  }
  __syncthreads();
  return result;
}

template <typename scalar_t>
__global__ void softmax_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ y, int64_t cols) {
  const int64_t row = blockIdx.x;
  const scalar_t* x_row = x + row * cols;
  scalar_t* y_row = y + row * cols;

  MaxSum state{-INFINITY, 0.f};
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    state = combine(state, MaxSum{static_cast<float>(x_row[c]), 1.f});
  }
  state = block_reduce(state);

  const float inv_sum = 1.f / state.d;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    y_row[c] = static_cast<scalar_t>(expf(static_cast<float>(x_row[c]) - state.m) * inv_sum);
  }
}

// Enough threads that each handles ~4 elements, between one warp and 1024.
int threads_for(int64_t cols) {
  int threads = 32;
  while (threads < 1024 && threads * 4 < cols) threads *= 2;
  return threads;
}

}  // namespace

torch::Tensor forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "softmax: x must be a CUDA tensor");
  TORCH_CHECK(x.dim() >= 1, "softmax: x must have at least one dimension");

  auto xc = x.contiguous();
  auto y = torch::empty_like(xc);
  const int64_t cols = xc.size(-1);
  if (xc.numel() == 0) return y;
  const int64_t rows = xc.numel() / cols;

  const at::cuda::CUDAGuard guard(xc.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = threads_for(cols);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, xc.scalar_type(),
                                  "softmax_forward", [&] {
    softmax_kernel<scalar_t><<<rows, threads, 0, stream>>>(
        xc.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), cols);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
