
#ifndef GRAPHLEARN_TORCH_CUDA_INDUCER_CUH_
#define GRAPHLEARN_TORCH_CUDA_INDUCER_CUH_


#include <limits>
#include <torch/script.h>
// #include "graphlearn_torch/include/sparse_matrix.h"

namespace graphlearn_torch {

torch::Tensor GeSpMM(
  // const c10::intrusive_ptr<SparseMatrix>& sparse_mat,
  torch::Tensor indptr,
  torch::Tensor indices,
  torch::Tensor values,
  std::vector<int64_t> shape,
torch::Tensor dense_mat);

}  // namespace graphlearn_torch

#endif  //  GRAPHLEARN_TORCH_CUDA_INDUCER_CUH_
