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
 * Copyright (c) 2017 by Contributors
 * \file quantized_activation.cc
 * \brief
*/
#include "../tensor/elemwise_unary_op.h"
#include "../nn/activation-inl.h"
#if MXNET_USE_MKLDNN == 1
#include "../nn/mkldnn/mkldnn_ops-inl.h"
#include "../nn/mkldnn/mkldnn_base-inl.h"
#endif
#include <vector>
#include <mxnet/operator_util.h>
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

bool QuantizedActivationShape(const nnvm::NodeAttrs& attrs,
                              std::vector<TShape>* in_shape,
                              std::vector<TShape>* out_shape) {
  CHECK_EQ(in_shape->size(), 3U);

  CHECK(!shape_is_none(in_shape->at(0)));
  for (int i = 1; i < 3; ++i) {
    SHAPE_ASSIGN_CHECK(*in_shape, i, TShape{1});
  }

  out_shape->clear();
  out_shape->push_back(in_shape->at(0));
  out_shape->push_back(TShape{1});
  out_shape->push_back(TShape{1});
  return true;
}

bool QuantizedActivationType(const nnvm::NodeAttrs& attrs,
                             std::vector<int> *in_type,
                             std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 3U);
  CHECK_EQ((*in_type)[0], mshadow::kInt8)
    << "`quantized_activation` only supports int8 input for now";
  for (int i = 1; i < 3; ++i) {
    TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kFloat32);
  }

  out_type->clear();
  out_type->push_back(mshadow::kInt8);
  out_type->push_back(mshadow::kFloat32);
  out_type->push_back(mshadow::kFloat32);
  return true;
}

inline static bool QuantizedActivationStorageType(const nnvm::NodeAttrs& attrs,
                                                  const int dev_mask,
                                                  DispatchMode* dispatch_mode,
                                                  std::vector<int> *in_attrs,
                                                  std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3);
  CHECK_EQ(out_attrs->size(), 3);
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  bool ret = ElemwiseStorageType<3, 3, false, false, false>(attrs, dev_mask,
                                                            dispatch_mode,
                                                            in_attrs, out_attrs);
#if MXNET_USE_MKLDNN == 1
  if (dev_mask == mshadow::cpu::kDevMask && SupportMKLDNNAct(param)) {
    *dispatch_mode = DispatchMode::kFComputeEx;
  }
#endif
  return ret;
}

void QuantizedActivationComputeCPU(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<NDArray>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<NDArray>& outputs) {
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 3U);
#if MXNET_USE_MKLDNN == 1
  MKLDNNQuantizedActivationForward(attrs, ctx, inputs[0], req[0], outputs[0]);
#endif
}

NNVM_REGISTER_OP(_contrib_quantized_activation)
.describe(R"code(Applies an activation function element-wise to the input.
)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(3)
.set_attr_parser(ParamParser<ActivationParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "min_data", "max_data"};
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output", "min_output", "max_output"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", QuantizedActivationShape)
.set_attr<nnvm::FInferType>("FInferType", QuantizedActivationType)
.set_attr<FInferStorageType>("FInferStorageType", QuantizedActivationStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", QuantizedActivationComputeCPU)
.add_argument("data", "NDArray-or-Symbol", "Input array to activation function.")
.add_argument("min_data", "NDArray-or-Symbol", "")
.add_argument("max_data", "NDArray-or-Symbol", "");

NNVM_REGISTER_OP(Activation)
.set_attr<FQuantizedOp>("FQuantizedOp", [](const NodeAttrs& attrs) {
    nnvm::NodePtr node = nnvm::Node::Create();
    node->attrs.op = Op::Get("_contrib_quantized_activation");
    node->attrs.name = "quantized_" + attrs.name;
    node->attrs.dict = attrs.dict;
    if (node->op()->attr_parser != nullptr) {
      node->op()->attr_parser(&(node->attrs));
    }
    return node;
  });

}  // namespace op
}  // namespace mxnet
