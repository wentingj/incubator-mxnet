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
 * \file mkldnn_quantized_conv-inl.h
 * \author Wenting Jiang
 * \brief
 */

#ifndef MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_DEQUANTIZE_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_DEQUANTIZE_INL_H_
#if MXNET_USE_MKLDNN == 1
#include <string>
#include <algorithm>
#include <vector>
#include "../../nn/mkldnn/mkldnn_base-inl.h"

namespace mxnet {
namespace op {

template<typename SrcType, typename DstType>
void MKLDNNDequantizeComputeKer(const std::vector<TBlob>& inputs,
                                const std::vector<TBlob>& outputs) {
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

  float quantized_range = MaxAbs(MaxValue<SrcType>(), MinValue<SrcType>());
  float real_range = MaxAbs(*inputs[1].dptr<DstType>(), *inputs[2].dptr<DstType>());
  if (inputs[0].type_flag_ == mshadow::kInt8) {
    quantized_range = MinAbs(MaxValue<SrcType>(), MinValue<SrcType>());
    real_range = MaxAbs(*inputs[1].dptr<DstType>(), *inputs[2].dptr<DstType>());
  }
  float scale = real_range / quantized_range;

  primitive_attr attr;
  const int mask = 0;
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
  auto reorder_pd = reorder::primitive_desc(i_mpd, o_mpd, attr);
  auto input = memory(i_mpd, inputs[0].dptr<SrcType>());
  auto output = memory(o_mpd, outputs[0].dptr<DstType>());
  auto r = reorder(reorder_pd, input, output);
  stream(stream::kind::lazy).submit({r}).wait();
}

void MKLDNNDequantizeCompute(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  if (inputs[0].type_flag_ == mshadow::kUint8) {
    MKLDNNDequantizeComputeKer<uint8_t, float>(inputs, outputs);
  } else if (inputs[0].type_flag_ == mshadow::kInt8) {
    MKLDNNDequantizeComputeKer<int8_t, float>(inputs, outputs);
  } else {
    LOG(FATAL) << "dequantize op only supports int8 and uint8 as input type";
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_DEQUANTIZE_INL_H_
