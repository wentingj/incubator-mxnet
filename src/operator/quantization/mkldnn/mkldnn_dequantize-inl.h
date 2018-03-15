/*******************************************************************************
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
*
* \file mkldnn_quantized_conv-inl.h
* \brief
*
*******************************************************************************/
#ifndef MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_DEQUANTIZE_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_DEQUANTIZE_INL_H_
#include <string>
#include <algorithm>
#include <vector>
#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../dequantize-inl.h"

namespace mxnet {
namespace op {

template<typename SrcType, typename DstType>
void MKLDequantizeComputeUnsigned(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  using red::limits::MaxValue;
  using red::limits::MinValue;
  Stream<cpu> *s = ctx.get_stream<cpu>();

  const DequantizeParam& param = nnvm::get<DequantizeParam>(attrs.parsed);

  // check shapes
  int i_dim = inputs[0].ndim();
  int o_dim = outputs[0].ndim();

  CHECK_EQ(i_dim, o_dim);

  int total_len = 1;

  memory::dims tensor_shape;

  for (size_t ii = 0; ii < i_dim; ++ii) {
    CHECK_EQ(inputs[0].size(ii), outputs[0].size(ii));
    total_len *= inputs[0].size(ii);
  }
  tensor_shape.push_back(total_len);

  float quantized_range = MaxAbs(MaxValue<SrcType>(), MinValue<SrcType>());
  float real_range = MaxAbs(*inputs[1].dptr<DstType>(), *inputs[2].dptr<DstType>());
  //std::cout<<"--MKLDeuantizeComputeKerUint8: real_range="<<real_range<<std::endl;
  //std::cout<<"                       : inputs[1]="<<inputs[1].dptr<DstType>()[0]<<std::endl;
  //std::cout<<"                       : inputs[2]="<<inputs[2].dptr<DstType>()[0]<<std::endl;
  //std::cout<<"                       : quantized_range="<<quantized_range<<std::endl;
  //std::cout<<"                       : real_range="<<real_range<<std::endl;

  float scale = real_range / quantized_range;

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

template<typename SrcType, typename DstType>
void MKLDequantizeComputeKer(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  using red::limits::MaxValue;
  using red::limits::MinValue;
  Stream<cpu> *s = ctx.get_stream<cpu>();

  const DequantizeParam& param = nnvm::get<DequantizeParam>(attrs.parsed);

  // check shapes
  int i_dim = inputs[0].ndim();
  int o_dim = outputs[0].ndim();

  CHECK_EQ(i_dim, o_dim);

  int total_len = 1;

  memory::dims tensor_shape;

  for (size_t ii = 0; ii < i_dim; ++ii) {
    CHECK_EQ(inputs[0].size(ii), outputs[0].size(ii));
    total_len *= inputs[0].size(ii);
  }
  tensor_shape.push_back(total_len);

  float quantized_range = MinAbs(MaxValue<SrcType>(), MinValue<SrcType>());
  float real_range = MaxAbs(*inputs[1].dptr<DstType>(), *inputs[2].dptr<DstType>());
  std::cout<<"--MKLDeuantizeComputeKerInt8: real_range="<<real_range<<std::endl;
  std::cout<<"                       : inputs[1]="<<inputs[1].dptr<DstType>()[0]<<std::endl;
  std::cout<<"                       : inputs[2]="<<inputs[2].dptr<DstType>()[0]<<std::endl;
  std::cout<<"                       : quantized_range="<<quantized_range<<std::endl;
  std::cout<<"                       : real_range="<<real_range<<std::endl;

  float scale = real_range / quantized_range;

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

void MKLDequantizeCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {

   if (inputs[0].type_flag_ == mshadow::kInt8) {
     MKLDequantizeComputeKer<int8_t, float>(attrs,
                                         ctx,
                                         inputs,
                                         req,
                                         outputs
                                         );
   } else if (inputs[0].type_flag_ == mshadow::kUint8) {
     MKLDequantizeComputeUnsigned<uint8_t, float>(attrs,
                                          ctx,
                                          inputs,
                                          req,
                                          outputs
                                          );
   }
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_DEQUANTIZE_INL_H_
