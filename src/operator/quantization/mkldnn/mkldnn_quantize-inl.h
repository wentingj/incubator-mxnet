/*
 * Copyright 2016-2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file mkldnn_quantize-inl.h
 * \brief
 * \author Wenting Jiang
 */
#ifndef MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZE_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZE_INL_H_
#include <string>
#include <algorithm>
#include <vector>
#include "../quantize-inl.h"
#include "../../nn/mkldnn/mkldnn_base-inl.h"

namespace mxnet {
namespace op {

template<typename SrcType, typename DstType>
void MKLDNNQuantizeComputeKer(const std::vector<TBlob>& inputs,
                              const std::vector<TBlob>& outputs,
                              const QuantizeParam& param) {
  using namespace mshadow;
  using namespace mxnet_op;
  using red::limits::MaxValue;
  using red::limits::MinValue;
  // check shapes
  size_t i_dim = inputs[0].ndim();
  size_t o_dim = outputs[0].ndim();
  CHECK_EQ(i_dim, o_dim);
  int total_len = 1;
  memory::dims tensor_shape;
  for (size_t i = 0; i < i_dim; ++i) {
    CHECK_EQ(inputs[0].size(i), outputs[0].size(i));
    total_len *= inputs[0].size(i);
  }
  tensor_shape.push_back(total_len);
  float real_range = MaxAbs(*inputs[1].dptr<float>(), *inputs[2].dptr<float>());
  float quantized_range = MaxAbs(MaxValue<DstType>(), MinValue<DstType>());
  *outputs[1].dptr<float>() = *inputs[1].dptr<float>();
  *outputs[2].dptr<float>() = *inputs[2].dptr<float>();
  if (param.out_type == mshadow::kInt8) {
    real_range = MaxAbs(*inputs[1].dptr<float>(), *inputs[2].dptr<float>());
    quantized_range = MinAbs(MaxValue<DstType>(), MinValue<DstType>());
    *outputs[1].dptr<float>() = -real_range;
    *outputs[2].dptr<float>() = real_range;
  }
  float scale = quantized_range / real_range;
  
  primitive_attr attr;
  int mask = 0;
  std::vector<float> scales = {scale};
  attr.set_output_scales(mask, scales);
  attr.set_int_output_round_mode(round_nearest);
  mkldnn::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();
  auto i_mpd = memory::primitive_desc({tensor_shape,
                                      (mkldnn::memory::data_type)data_type_enum<SrcType>::type,
                                       memory::format::x},
                                       cpu_engine);
  auto o_mpd = memory::primitive_desc({tensor_shape,
                                      (mkldnn::memory::data_type)data_type_enum<DstType>::type,
                                       memory::format::x},
                                       cpu_engine);
  auto reorder_pd  = reorder::primitive_desc(i_mpd, o_mpd, attr);
  auto input = memory(i_mpd, inputs[0].dptr<SrcType>());
  auto output = memory(o_mpd, outputs[0].dptr<DstType>());
  auto r = reorder(reorder_pd, input, output);
  stream(stream::kind::lazy).submit({r}).wait();
}

void MKLDNNQuantizeCompute(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  const QuantizeParam& param = nnvm::get<QuantizeParam>(attrs.parsed);
  if (param.out_type == mshadow::kUint8) {
    MKLDNNQuantizeComputeKer<float, uint8_t>(inputs, outputs, param);
  } else if (param.out_type == mshadow::kInt8) {
    MKLDNNQuantizeComputeKer<float, int8_t>(inputs, outputs, param);
  } else {
    LOG(FATAL) << "quantize op only supports int8 and uint8 as output type";
  }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MKL_DNN_MKLDNN_QUANTIZE_INL_H_

