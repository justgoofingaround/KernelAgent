// RMSNorm, one thread block per row: y = x * rsqrt(mean(x^2) + eps) * w.
//
// When the row width and all pointers allow it, each thread moves 128 bits per
// load/store (4 x fp32 or 8 x fp16/bf16), which cuts the number of memory
// instructions and helps reach DRAM bandwidth. Otherwise it falls back to
// scalar accesses. Sum of squares is accumulated in fp32.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

namespace {

constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kVecBytes = 16;

template <typename T, int N>
struct alignas(sizeof(T) * N) Vec {
  T val[N];
};

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(kFullMask, v, offset);
  return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
  __shared__ float partials[32];
  __shared__ float result;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;

  v = warp_reduce_sum(v);
  if (lane == 0) partials[warp] = v;
  __syncthreads();

  if (warp == 0) {
    v = lane < num_warps ? partials[lane] : 0.f;
    v = warp_reduce_sum(v);
    if (lane == 0) result = v;
  }
  __syncthreads();
  return result;
}

// VEC = elements per memory access (1 = scalar fallback).
template <typename scalar_t, int VEC>
__global__ void rmsnorm_kernel(const scalar_t* __restrict__ x, const scalar_t* __restrict__ w,
                               scalar_t* __restrict__ y, int64_t cols, float eps) {
  using V = Vec<scalar_t, VEC>;
  const int64_t row = blockIdx.x;
  const V* x_row = reinterpret_cast<const V*>(x + row * cols);
  const V* w_vec = reinterpret_cast<const V*>(w);
  V* y_row = reinterpret_cast<V*>(y + row * cols);
  const int64_t num_vecs = cols / VEC;

  float sum_sq = 0.f;
  for (int64_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    const V v = x_row[i];
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      const float f = static_cast<float>(v.val[k]);
      sum_sq += f * f;
    }
  }
  const float inv_rms = rsqrtf(block_reduce_sum(sum_sq) / static_cast<float>(cols) + eps);

  // Second read of the row usually hits L1/L2: the block just loaded it.
  for (int64_t i = threadIdx.x; i < num_vecs; i += blockDim.x) {
    const V v = x_row[i];
    const V wv = w_vec[i];
    V out;
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      out.val[k] = static_cast<scalar_t>(static_cast<float>(v.val[k]) * inv_rms *
                                         static_cast<float>(wv.val[k]));
    }
    y_row[i] = out;
  }
}

bool is_aligned(const void* ptr, int bytes) {
  return reinterpret_cast<std::uintptr_t>(ptr) % bytes == 0;
}

int threads_for(int64_t work_items) {
  int threads = 32;
  while (threads < 1024 && threads * 4 < work_items) threads *= 2;
  return threads;
}

template <typename scalar_t, int VEC>
void launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y, int64_t rows,
            int64_t cols, float eps, cudaStream_t stream) {
  rmsnorm_kernel<scalar_t, VEC><<<rows, threads_for(cols / VEC), 0, stream>>>(
      x.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), cols, eps);
}

}  // namespace

torch::Tensor forward(torch::Tensor x, torch::Tensor w, double eps) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "rmsnorm: x and w must be CUDA tensors");
  TORCH_CHECK(x.dim() >= 1, "rmsnorm: x must have at least one dimension");
  TORCH_CHECK(w.dim() == 1 && w.size(0) == x.size(-1), "rmsnorm: w must have shape (hidden,)");
  TORCH_CHECK(x.scalar_type() == w.scalar_type(), "rmsnorm: x and w must have the same dtype");

  auto xc = x.contiguous();
  auto wc = w.contiguous();
  auto y = torch::empty_like(xc);
  const int64_t cols = xc.size(-1);
  if (xc.numel() == 0) return y;
  const int64_t rows = xc.numel() / cols;

  const at::cuda::CUDAGuard guard(xc.device());
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, xc.scalar_type(),
                                  "rmsnorm_forward", [&] {
    constexpr int vec = kVecBytes / sizeof(scalar_t);
    const bool can_vectorize = cols % vec == 0 && is_aligned(xc.data_ptr(), kVecBytes) &&
                               is_aligned(wc.data_ptr(), kVecBytes) &&
                               is_aligned(y.data_ptr(), kVecBytes);
    if (can_vectorize) {
      launch<scalar_t, vec>(xc, wc, y, rows, cols, static_cast<float>(eps), stream);
    } else {
      launch<scalar_t, 1>(xc, wc, y, rows, cols, static_cast<float>(eps), stream);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
