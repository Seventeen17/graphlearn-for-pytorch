/* Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #include "graphlearn_torch/csrc/cuda/gespmm.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include "graphlearn_torch/include/common.cuh"
#include <torch/script.h>
// #include "graphlearn_torch/include/sparse_matrix.h"

namespace graphlearn_torch {

namespace {
constexpr int32_t WARP_SIZE = 128;
constexpr int32_t BLOCK_WARPS = 1;
// The number of rows covered by each threadblock.
constexpr int32_t TILE_SIZE = BLOCK_WARPS;

template <typename scalar_t>
__global__ void SpMMKernel(
    const int64_t* __restrict__ indptr,
    const int64_t* __restrict__ indices,
    const scalar_t* __restrict__ ufeat,
    scalar_t* __restrict__ out,
    int num_row,
    int num_cols,
    int element_size) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid >= element_size) return;

  // TODO, this is fake now.
  out[tid] = 0.0;
}

torch::Tensor GeSpMM(
    // const c10::intrusive_ptr<SparseMatrix>& sparse_mat,
    torch::Tensor indptr,
    torch::Tensor indices,
    torch::Tensor values,
    std::vector<int64_t> shape,
    torch::Tensor dense_mat) {
  std::vector<int64_t> out_shape = {shape[0], dense_mat.size(1)};
  auto indptr_ptr = indptr.data_ptr<int64_t>();
  auto indices_ptr = indices.data_ptr<int64_t>();
  int num_rows = out_shape[0];
  int num_cols = out_shape[1];
  int element_size = num_rows * num_cols;
  auto out = dense_mat.new_empty({num_rows, num_cols});

  // TODO
  const dim3 block(num_rows);
  const dim3 grid((element_size + num_cols - 1) / num_cols);

  cudaStream_t stream = ::at::cuda::getDefaultCUDAStream();

  AT_DISPATCH_ALL_TYPES(dense_mat.scalar_type(), "SpMMKernel", [&] {
    const auto ufeat_data = dense_mat.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    SpMMKernel<scalar_t>
      <<<grid, block, 0, stream>>>(
        indptr_ptr, indices_ptr, ufeat_data, out_data,
        num_rows, num_cols, element_size);
  });
  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(graphlearn_torch, CUDA, m) {
  m.impl("GeSpMM", GeSpMM);
}

} // namespace graphlearn_torch
