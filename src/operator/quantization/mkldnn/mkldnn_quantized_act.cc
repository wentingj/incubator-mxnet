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
 * \file mkldnn_quantized_act.cc
*/

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../../operator_common.h"
#include "../../nn/activation-inl.h"
#include "../../nn/mkldnn/mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1

#include <mkldnn.hpp>

namespace mxnet {
namespace op {

static inline mkldnn::algorithm GetMKLDNNActAlgo(const ActivationParam& param) {
  switch (param.act_type) {
    case activation::kReLU:
      return mkldnn::algorithm::eltwise_relu;
    case activation::kSigmoid:
      return mkldnn::algorithm::eltwise_logistic;
    case activation::kTanh:
      return mkldnn::algorithm::eltwise_tanh;
    case activation::kSoftReLU:
      return mkldnn::algorithm::eltwise_soft_relu;
    default:
      LOG(FATAL) << "unknown activation type";
      return mkldnn::algorithm::eltwise_relu;
  }
}

typedef std::shared_ptr<mkldnn::eltwise_forward::primitive_desc> mkldnn_act_pdesc_ptr;

static mkldnn::eltwise_forward::primitive_desc GetActFwdDescImpl(
    const ActivationParam& param, bool is_train,
    const mkldnn::memory &input_mem, int dtype) {
  mkldnn::memory::primitive_desc data_mpd = input_mem.get_primitive_desc();
  mkldnn::memory::desc data_md = data_mpd.desc();
  auto cpu_engine = data_mpd.get_engine();

  auto alg = GetMKLDNNActAlgo(param);
  MSHADOW_TYPE_SWITCH(dtype, DType, {
    DType alpha = 0;
    mkldnn::eltwise_forward::desc desc = is_train
        ? mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward_training,
                                        alg, data_md, alpha)
        : mkldnn::eltwise_forward::desc(mkldnn::prop_kind::forward_scoring,
                                        alg, data_md, alpha);
    return mkldnn::eltwise_forward::primitive_desc(desc, cpu_engine);
  });
  LOG(INFO) << "Unsupported data type for MKLDNN activation";
  mkldnn::eltwise_forward::desc desc = mkldnn::eltwise_forward::desc(
      mkldnn::prop_kind::forward_training, alg, data_md, 0);
  return mkldnn::eltwise_forward::primitive_desc(desc, cpu_engine);
}

typedef MKLDNNParamOpSign<ActivationParam> MKLDNNActSignature;

class MKLDNNActForward {
  std::shared_ptr<mkldnn::eltwise_forward> fwd;
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> out;

 public:
  const mkldnn::eltwise_forward::primitive_desc fwd_pd;

  MKLDNNActForward(const ActivationParam& param, bool is_train,
                   const NDArray &data, const mkldnn::memory &mem): fwd_pd(
                       GetActFwdDescImpl(param, is_train, mem, data.dtype())) {
  }

  void SetNewMem(const mkldnn::memory &data, const mkldnn::memory &output) {
    if (this->data == nullptr)
      this->data = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              data.get_primitive_desc(), data.get_data_handle()));
    else
      this->data->set_data_handle(data.get_data_handle());

    CHECK(fwd_pd.dst_primitive_desc() == output.get_primitive_desc());
    if (this->out == nullptr)
      this->out = std::shared_ptr<mkldnn::memory>(new mkldnn::memory(
              fwd_pd.dst_primitive_desc(), output.get_data_handle()));
    else
      this->out->set_data_handle(output.get_data_handle());

    if (this->fwd == nullptr) {
      this->fwd = std::shared_ptr<mkldnn::eltwise_forward>(
          new mkldnn::eltwise_forward(fwd_pd, mkldnn::primitive::at(*this->data),
                                      *this->out));
    }
  }

  const mkldnn::eltwise_forward &GetFwd() const {
    return *fwd;
  }
};

static MKLDNNActForward &GetActForward(const ActivationParam& param,
                                       const OpContext &ctx, const NDArray &in_data,
                                       const mkldnn::memory &in_mem) {
  static thread_local std::unordered_map<MKLDNNActSignature, MKLDNNActForward, MKLDNNOpHash> fwds;
  MKLDNNActSignature key(param);
  key.AddSign(ctx.is_train);
  key.AddSign(param.act_type);
  key.AddSign(in_data);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNActForward fwd(param, ctx.is_train, in_data, in_mem);
    auto ins_ret = fwds.insert(std::pair<MKLDNNActSignature, MKLDNNActForward>(
            key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNQuantizedActivationForward(const nnvm::NodeAttrs& attrs,
                                      const OpContext &ctx,
                                      const NDArray &in_data,
                                      const OpReqType &req,
                                      const NDArray &out_data) {
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  auto input_mem = in_data.GetMKLDNNData();
  MKLDNNActForward &fwd = GetActForward(param, ctx, in_data, *input_mem);
  auto out_mem = const_cast<NDArray &>(out_data).CreateMKLDNNData(
      fwd.fwd_pd.dst_primitive_desc());
  fwd.SetNewMem(*input_mem, *out_mem);
  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrim(fwd.GetFwd());
  stream->Submit();
}

}  // namespace op
}  // namespace mxnet

#endif
