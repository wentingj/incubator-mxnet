/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file mkldnn_quantized_conv.cc
*/

#include "../../nn/convolution-inl.h"
#include "../../nn/mkldnn/mkldnn_ops-inl.h"
#include "../../nn/mkldnn/mkldnn_base-inl.h"
#include "../quantization_utils.h"
#include "../../tensor/matrix_op-inl.h"
#include "../../elemwise_op_common.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

static mkldnn::convolution_forward::primitive_desc GetConvFwdImpl(
    const ConvolutionParam& param, bool is_train, const NDArray &data,
    const NDArray &weights, const NDArray *bias, const NDArray &output) {
  auto prop = is_train ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;
  
  //mkldnn::memory::dims dims(data.shape().ndim());
  //for (size_t i = 0; i < dims.size(); i++) dims[i] = data.shape()[i];
  //auto data_md = mkldnn::memory::desc{dims, 
  //                                    (mkldnn::memory::data_type)data_type_enum<uint8_t>::type,
  //                                    mkldnn::memory::format::any};
  auto data_md = GetMemDesc(data);
  
  auto weight_md = GetWeightDesc(weights, param.num_group);
  auto out_md = GetMemDesc(output);
  auto engine = CpuEngine::Get()->get_engine();
  mkldnn::memory::dims strides{0, 0};
  if (param.stride.ndim() == 2) {
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
  }
  mkldnn::memory::dims padding{0, 0};
  if (param.pad.ndim() == 2) {
    padding[0] = param.pad[0];
    padding[1] = param.pad[1];
  }
  mkldnn::primitive_attr attr;
  int mask = 0;
  int output_shift = 0;
  float scale = pow(2. ,output_shift);
  std::vector<float> scales = {scale};
  attr.set_output_scales(mask,scales);
  attr.set_int_output_round_mode(mkldnn::round_nearest);
  
  if (param.dilate.ndim() == 0 && bias == nullptr) {
    mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
        data_md, weight_md, out_md, strides, padding, padding, mkldnn::padding_kind::zero);
    return mkldnn::convolution_forward::primitive_desc(desc, attr, engine);
  } else if (param.dilate.ndim() == 0) {
    auto bias_md = GetMemDesc(*bias);
    mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
        data_md, weight_md, bias_md, out_md, strides, padding, padding,
        mkldnn::padding_kind::zero);
    return mkldnn::convolution_forward::primitive_desc(desc, attr, engine);
  } else {
    mkldnn::memory::dims dilates{0, 0};
    if (param.dilate.ndim() == 2) {
      dilates[0] = param.dilate[0] - 1;
      dilates[1] = param.dilate[1] - 1;
    }
    if (bias == nullptr) {
      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
          data_md, weight_md, out_md, strides, dilates, padding, padding,
          mkldnn::padding_kind::zero);
      return mkldnn::convolution_forward::primitive_desc(desc, attr, engine);
    } else {
      auto bias_md = GetMemDesc(*bias);
      mkldnn::convolution_forward::desc desc(prop, mkldnn::algorithm::convolution_direct,
                                             data_md, weight_md, bias_md, out_md, strides,
                                             dilates, padding, padding,
                                             mkldnn::padding_kind::zero);
      return mkldnn::convolution_forward::primitive_desc(desc, attr, engine);
    }
  }
}

class MKLDNNConvForward {
  std::shared_ptr<mkldnn::convolution_forward> fwd;
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> weight;
  std::shared_ptr<mkldnn::memory> bias;
  std::shared_ptr<mkldnn::memory> out;

 public:
  mkldnn::convolution_forward::primitive_desc fwd_pd;

  MKLDNNConvForward(const ConvolutionParam& param, bool is_train,
                    const NDArray &data, const NDArray &weights,
                    const NDArray *bias, const NDArray &output): fwd_pd(
                        GetConvFwdImpl(param, is_train, data, weights, bias, output)) {
  }

  void SetNewMem(const mkldnn::memory &data, const mkldnn::memory &weight,
                 const mkldnn::memory *bias, const mkldnn::memory &output) {
    if (this->data == nullptr)
      this->data = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              fwd_pd.src_primitive_desc(), data.get_data_handle()));
    else
      this->data->set_data_handle(data.get_data_handle());

    if (this->weight == nullptr)
      this->weight = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              fwd_pd.weights_primitive_desc(), weight.get_data_handle()));
    else
      this->weight->set_data_handle(weight.get_data_handle());

    if (this->out == nullptr)
      this->out = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              fwd_pd.dst_primitive_desc(), output.get_data_handle()));
    else
      this->out->set_data_handle(output.get_data_handle());

    if (bias != nullptr) {
      if (this->bias == nullptr)
        this->bias = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
                fwd_pd.bias_primitive_desc(), bias->get_data_handle()));
      else
        this->bias->set_data_handle(bias->get_data_handle());
      if (this->fwd == nullptr)
        this->fwd = std::shared_ptr<mkldnn::convolution_forward>(
            new mkldnn::convolution_forward(fwd_pd, mkldnn::primitive::at(*this->data),
                                            mkldnn::primitive::at(*this->weight),
                                            mkldnn::primitive::at(*this->bias),
                                            *this->out));
    } else if (this->fwd == nullptr) {
      this->fwd = std::shared_ptr<mkldnn::convolution_forward>(
          new mkldnn::convolution_forward(fwd_pd, mkldnn::primitive::at(*this->data),
                                          mkldnn::primitive::at(*this->weight),
                                          *this->out));
    }
  }

  const mkldnn::convolution_forward &GetFwd() const {
    return *fwd;
  }
};

typedef MKLDNNParamOpSign<ConvolutionParam> MKLDNNConvSignature;

static inline MKLDNNConvForward &GetConvFwd(
    const nnvm::NodeAttrs& attrs, bool is_train,
    const NDArray &data, const NDArray &weights,
    const NDArray *bias, const NDArray &output) {
  static thread_local std::unordered_map<MKLDNNConvSignature, MKLDNNConvForward, MKLDNNOpHash> fwds;
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  MKLDNNConvSignature key(param);
  key.AddSign(is_train);
  // Here we can sign the conv op with NDArray because conv primitive will
  // decide the right layout for the, so we only need to get the shape and the
  // data type of the arrays.
  key.AddSign(data);
  key.AddSign(weights);
  key.AddSign(output);
  if (bias)
    key.AddSign(*bias);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNConvForward fwd(param, is_train, data, weights, bias, output);
    auto ins_ret = fwds.insert(
        std::pair<MKLDNNConvSignature, MKLDNNConvForward>(key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

const mkldnn::memory *MKLDNNReoder (const NDArray &input) {
  mkldnn::engine cpu_engine = mxnet::CpuEngine::Get()->get_engine();
  mxnet::TShape sh = input.shape();
  int i_dim = sh.ndim();
  int total_len = 1;
  memory::dims tensor_shape;
  for (size_t i = 0; i < i_dim; ++i) {
    total_len *= sh[i];
  }
  tensor_shape.push_back(total_len);
  
  primitive_attr attr;
  int mask = 0;
  std::vector<float> scales = {1.0};
  attr.set_output_scales(mask, scales);
  attr.set_int_output_round_mode(round_nearest);
 
  auto i_mpd = memory::primitive_desc({tensor_shape,
                                      (mkldnn::memory::data_type)data_type_enum<int8_t>::type,
                                       memory::format::x},
                                       cpu_engine);
  auto o_mpd = memory::primitive_desc({tensor_shape,
                                      (mkldnn::memory::data_type)data_type_enum<uint8_t>::type,
                                       memory::format::x},
                                       cpu_engine);
  auto o_mem = new mkldnn::memory(o_mpd);
  auto in_mem = memory(i_mpd, input.data().dptr<int8_t>());
  //auto o_mem = memory(o_mpd, input.data().dptr<uint8_t>());
  auto reorder_pd  = reorder::primitive_desc(i_mpd, o_mpd, attr);
  auto r = mkldnn::reorder(reorder_pd, in_mem, *o_mem);
  stream(stream::kind::lazy).submit({r}).wait();
  return o_mem;
}

void MKLDNNQuantized_conv2dForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                               const std::vector<NDArray> &in_data,
                               const std::vector<OpReqType> &req,
                               const std::vector<NDArray> &out_data) {
  Stream<cpu> *s = ctx.get_stream<cpu>();
  
  TmpMemMgr::Get()->Init(ctx.requested[conv::kTempSpace]);
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  //auto in_mem = in_data[conv::kData].GetMKLDNNDataReorder(fwd.fwd_pd.src_primitive_desc());
  //auto in_mem = in_data[conv::kData].GetMKLDNNData();
  //auto data_mem = MKLDNNReoder(in_data[conv::kData]);//, *in_mem); 
 
  MKLDNNConvForward &fwd = GetConvFwd(attrs,
      ctx.is_train, in_data[0], in_data[conv::kWeight],
      param.no_bias ? nullptr : &in_data[conv::kBias], out_data[conv::kOut]);
  
  auto data_mem = in_data[conv::kData].GetMKLDNNDataReorder(fwd.fwd_pd.src_primitive_desc()); 
  //auto data_mem = ret.GetMKLDNNDataReorder(fwd.fwd_pd.src_primitive_desc());
  auto weight_mem = GetWeights(in_data[conv::kWeight], fwd.fwd_pd.weights_primitive_desc(),
                               param.num_group);
  auto out_mem = CreateMKLDNNMem(out_data[conv::kOut], fwd.fwd_pd.dst_primitive_desc(),
                                 req[conv::kOut]);
  const mkldnn::memory *bias_mem = nullptr;
  if (!param.no_bias)
    bias_mem = in_data[conv::kBias].GetMKLDNNDataReorder(fwd.fwd_pd.bias_primitive_desc());
  fwd.SetNewMem(*data_mem, *weight_mem, bias_mem, *out_mem.second);
  MKLDNNStream::Get()->RegisterPrim(fwd.GetFwd());

  CommitOutput(out_data[conv::kOut], out_mem);
  MKLDNNStream::Get()->Submit();
  const size_t num_inputs = param.no_bias ? 2 : 3;
  mxnet_op::Kernel<QuantizationRangeForMultiplicationStruct, cpu>::Launch(s, 1,
           out_data[1].data().dptr<float>(), out_data[2].data().dptr<float>(),
           in_data[num_inputs].data().dptr<float>(),
           in_data[num_inputs+1].data().dptr<float>(),
           in_data[num_inputs+2].data().dptr<float>(),
           in_data[num_inputs+3].data().dptr<float>());
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
