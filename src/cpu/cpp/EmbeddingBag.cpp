/*****************************************************************************
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
 * All rights reserved.
 ******************************************************************************/

#include "EmbeddingUtils.hpp"
#include "EnvReader.hpp"

using namespace zendnnl::interface;

namespace zentorch {
at::Tensor
zendnnl_embeddingbag_impl(const at::Tensor &weight, const at::Tensor &indices,
                          const at::Tensor &offsets, bool scale_grad_by_freq,
                          int64_t mode, bool sparse,
                          c10::optional<at::Tensor> per_sample_weights_opt,
                          bool include_last_offset, int64_t padding_idx,
                          std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  zen_embedding_weight_check(weight);

  int dim_embedding = weight.sizes()[1];
  int num_bags = offsets.sizes()[0];

  if (include_last_offset == true) {
    num_bags -= 1;
  }

  LOG(INFO) << "Embedding matrix dimensions: " << weight.sizes()[0] << "x"
            << dim_embedding;
  LOG(INFO) << "Number of embedding bags: " << num_bags;

  c10::MaybeOwned<at::Tensor> per_sample_weights_opt_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  [[maybe_unused]] const at::Tensor &per_sample_weights =
      *per_sample_weights_opt_maybe_owned;
  auto per_sample_weights_defined = per_sample_weights.defined();

  at::Tensor output = at::detail::empty_strided_cpu(
      {num_bags, dim_embedding}, {dim_embedding, 1}, weight.options());

  const int int_env_value =
      EnvReader::getEnvVariableAsInt("USE_ZENDNN_EMBBAG_DIRECT");
  const bool use_zendnnl_direct_kernel = static_cast<bool>(int_env_value);
  if (use_zendnnl_direct_kernel) {
    // Build embag_params_t structure
    zendnnl::lowoha::embag::embag_params_t params;

    // Set data types
    params.dtypes.table = get_zendnnl_dtype(weight);
    params.dtypes.output = get_zendnnl_dtype(output);
    params.dtypes.indices = get_zendnnl_dtype(indices);
    params.dtypes.offsets = get_zendnnl_dtype(offsets);

    // Use algo directly (embag_algo_t is aliased to ops::embag_algo_t)
    switch (mode) {
    case EMBEDDING_BAG_ALGO::SUM:
      params.algo = embag_algo_t::sum;
      break;
    case EMBEDDING_BAG_ALGO::MEAN:
      params.algo = embag_algo_t::mean;
      break;
    case EMBEDDING_BAG_ALGO::MAX:
      params.algo = embag_algo_t::max;
      break;
    default:
      // Assigning sum as the default algorithm as that is seen in
      // native_functions.yaml file in Pytorch
      // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
      // func: embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool
      // scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor?
      // per_sample_weights=None, bool include_last_offset=False)
      LOG(WARNING) << "Invalid embedding bag algorithm: " << mode
                   << ". Using default algorithm: sum";
      params.algo = embag_algo_t::sum;
      break;
    }

    // Set dimensions
    params.num_embeddings = weight.sizes()[0];
    params.embedding_dim = weight.sizes()[1];
    params.num_indices = indices.sizes()[0];
    params.num_bags =
        include_last_offset ? offsets.sizes()[0] - 1 : offsets.sizes()[0];
    params.is_weights = per_sample_weights_defined;
    params.include_last_offset = include_last_offset;
    params.padding_idx = padding_idx;
    params.num_threads = 0; // Use default (omp_get_max_threads)
    params.fp16_scale_bias = true;
    params.dst_stride = output.strides()[0];

    // Call LOWOHA embedding_bag_direct API
    status_t status = zendnnl::lowoha::embag::embedding_bag_direct(
        weight.data_ptr(), indices.data_ptr(), offsets.data_ptr(),
        per_sample_weights_defined ? per_sample_weights.data_ptr<float>()
                                   : nullptr,
        output.data_ptr(), params);
    ZENTORCH_CHECK(status == status_t::success,
                   "LOA-operator for quant embedding bag failed.");
    return output;
  }

  tensor_t table = tensor_t();
  set_zendnnl_tensor_attributes(weight, table, "table");

  tensor_t indices_tensor = tensor_t();
  set_zendnnl_tensor_attributes(indices, indices_tensor, "indices");

  tensor_t offsets_tensor = tensor_t();
  set_zendnnl_tensor_attributes(offsets, offsets_tensor, "offsets");

  tensor_t output_tensor = tensor_t();
  set_zendnnl_tensor_attributes(output, output_tensor, "output");

  [[maybe_unused]] tensor_t per_sample_weights_tensor = tensor_t();

  if (per_sample_weights_defined) {
    LOG(INFO) << "Using the per-sample weights tensor!";
    set_zendnnl_tensor_attributes(per_sample_weights, per_sample_weights_tensor,
                                  "per_sample_weights");
  }

  embag_context_t embedding_bag_context = embag_context_t();

  set_embedding_context_attributes(embedding_bag_context, table, mode,
                                   include_last_offset, padding_idx,
                                   per_sample_weights_defined);

  // define embedding bag operator
  embag_operator_t embedding_bag_operator = embag_operator_t();
  if (per_sample_weights_defined) {
    set_embedding_operator_attributes(embedding_bag_operator, zentorch_op_name,
                                      embedding_bag_context, indices_tensor,
                                      output_tensor, offsets_tensor,
                                      per_sample_weights_tensor);
  } else {
    set_embedding_operator_attributes(embedding_bag_operator, zentorch_op_name,
                                      embedding_bag_context, indices_tensor,
                                      output_tensor, offsets_tensor);
  }

  LOG(INFO) << "EmbeddingBag compute in progress...";
  status_t status = embedding_bag_operator.execute();

  ZENTORCH_CHECK(status == status_t::success, "operator ",
                 embedding_bag_operator.get_name(), " execution failed.");

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return output;
}

at::Tensor
zentorch_embedding_bag(const at::Tensor &weight, const at::Tensor &indices,
                       const at::Tensor &offsets, bool scale_grad_by_freq,
                       int64_t mode, bool sparse,
                       c10::optional<at::Tensor> per_sample_weights_opt,
                       bool include_last_offset, int64_t padding_idx,
                       std::string zentorch_op_name) {
  return zendnnl_embeddingbag_impl(weight, indices, offsets, scale_grad_by_freq,
                                   mode, sparse, per_sample_weights_opt,
                                   include_last_offset, padding_idx,
                                   zentorch_op_name);
}

std::vector<at::Tensor> zendnnl_horizontal_embedding_bag_group_impl(
    at::TensorList weight, at::TensorList indices, at::TensorList offsets,
    at::IntArrayRef scale_grad_by_freq, at::IntArrayRef mode,
    at::IntArrayRef sparse,
    c10::List<c10::optional<at::Tensor>> per_sample_weights_opt,
    at::IntArrayRef include_last_offset, at::IntArrayRef padding_idx,
    std::string zentorch_op_name) {

  LOG(INFO) << "[" << __FILE__ << ": " << __LINE__ << "] "
            << "Executing function: " << __FUNCTION__;

  int num_eb_ops = weight.size();
  std::vector<at::Tensor> output(num_eb_ops);

  at::parallel_for(0, num_eb_ops, 0, [&](int64_t start, int64_t end) {
    for (int i = start; i < end; i++) {
      output[i] = zentorch_embedding_bag(
          weight[i], indices[i], offsets[i], scale_grad_by_freq[i], mode[i],
          sparse[i], per_sample_weights_opt[i], include_last_offset[i],
          padding_idx[i], zentorch_op_name);
    }
  });

  LOG(INFO) << "Finished executing: " << __FUNCTION__ << "!\n";

  return output;
}

std::vector<at::Tensor> zentorch_horizontal_embedding_bag_group(
    at::TensorList weight, at::TensorList indices, at::TensorList offsets,
    at::IntArrayRef scale_grad_by_freq, at::IntArrayRef mode,
    at::IntArrayRef sparse,
    c10::List<c10::optional<at::Tensor>> per_sample_weights_opt,
    at::IntArrayRef include_last_offset, at::IntArrayRef padding_idx,
    std::string zentorch_op_name) {

  return zendnnl_horizontal_embedding_bag_group_impl(
      weight, indices, offsets, scale_grad_by_freq, mode, sparse,
      per_sample_weights_opt, include_last_offset, padding_idx,
      zentorch_op_name);
}

TORCH_LIBRARY_FRAGMENT(zentorch, m) {
  m.def("zentorch_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, "
        "bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? "
        "per_sample_weights=None, bool include_last_offset=False, int "
        "padding_idx=-1, str "
        "zentorch_op_name='zentorch::zentorch_embedding_bag') -> Tensor");
  m.def("zentorch_horizontal_embedding_bag_group(Tensor[] weight, "
        "Tensor[] indices, Tensor[] offsets, int[] scale_grad_by_freq, "
        "int[] mode, int[] sparse, Tensor?[] per_sample_weights, "
        "int[] include_last_offset, int[] padding_idx, str "
        "zentorch_op_name = "
        "'zentorch::zentorch_horizontal_embedding_bag_group') -> Tensor[]");
}

TORCH_LIBRARY_IMPL(zentorch, CPU, m) {
  m.impl("zentorch_embedding_bag", zentorch_embedding_bag);
  m.impl("zentorch_horizontal_embedding_bag_group",
         zentorch_horizontal_embedding_bag_group);
}
} // namespace zentorch
