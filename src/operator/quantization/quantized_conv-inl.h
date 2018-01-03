/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_conv-inl.h
 * \brief
 * \author Ziheng Jiang, Jun Wu
*/
#ifndef MXNET_OPERATOR_CONTRIB_QUANTIZED_CONV_INL_H_
#define MXNET_OPERATOR_CONTRIB_QUANTIZED_CONV_INL_H_
#include <mxnet/operator.h>
#include <mxnet/op_attr_types.h>
#include "../operator_common.h"

namespace mxnet {
namespace op {

namespace qconv {
enum ConvolutionOpCudnnTune {kOff, kLimited, kFastest};
}

struct QuantizedConvParam :
  public dmlc::Parameter<QuantizedConvParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  TShape dilate;
  uint32_t num_filter;
  bool no_bias;
  int layout;
  int cudnn_tune;
  bool cudnn_off;
  uint32_t num_group;
  uint64_t workspace;
  DMLC_DECLARE_PARAMETER(QuantizedConvParam) {
    DMLC_DECLARE_FIELD(kernel);
    DMLC_DECLARE_FIELD(stride)
    .set_default(TShape())
    .describe("conv2d stride: (h, w)");
    DMLC_DECLARE_FIELD(pad)
    .set_default(TShape())
    .describe("pad for conv2d: (h, w)");
    DMLC_DECLARE_FIELD(dilate)
    .set_default(TShape())
    .describe("convolution dilate: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(num_filter);
    DMLC_DECLARE_FIELD(no_bias)
    .set_default(true);
    DMLC_DECLARE_FIELD(layout)
    .add_enum("NCHW", mshadow::kNCHW)
    .set_default(mshadow::kNCHW)
    .describe("Set layout for input, output and weight.");
    DMLC_DECLARE_FIELD(cudnn_tune)
    .add_enum("off", qconv::kOff)
    .add_enum("limited_workspace", qconv::kLimited)
    .add_enum("fastest", qconv::kFastest)
    .set_default(qconv::kOff)
    .describe("Whether to pick convolution algo by running performance test.");
    DMLC_DECLARE_FIELD(cudnn_off)
    .set_default(false)
    .describe("Turn off cudnn for this layer.");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
    .describe("Maximum temporary workspace allowed for convolution (MB).");
  }
};  // QuantizedConvParam

// TODO(junwu): Reuse the InferShape function of convolution op after
// this pr is merged: https://github.com/apache/incubator-mxnet/pull/8302
inline bool QuantizedConvShape(const nnvm::NodeAttrs& attrs,
                               std::vector<TShape>* in_shape,
                               std::vector<TShape>* out_shape) {
  using namespace mshadow;
  const QuantizedConvParam& param = nnvm::get<QuantizedConvParam>(attrs.parsed);
  CHECK_EQ(param.num_group, 1U) << "quantized_conv only supports num_group=1 for now";
  CHECK_EQ(in_shape->size(), param.no_bias? 6U : 9U);
  CHECK_EQ(param.layout, mshadow::kNCHW) << "quantized_conv only supports NCHW for now";
  CHECK_EQ(param.kernel.ndim(), 2U) << "quantized_conv only supports 2D convolution for now";
  CHECK(param.dilate.ndim() == 0U || param.dilate.Size() == 1U)
    << "quantized_conv only supports dilation=1 for all dimensions";
  const TShape& dshape =  in_shape->at(0);
  CHECK_EQ(dshape.ndim(), 4U);
  if (dshape.ndim() == 0U) return false;

  const int N = 0, H = 2, W = 3, C = 1;
  CHECK_EQ(dshape[C] % 4,  0U)
    << "for 8bit cudnn conv, the number of channel must be multiple of 4";
  CHECK_EQ(param.num_filter % 4, 0U)
    << "for 8bit cudnn conv, the number of channel must be multiple of 4";

  TShape wshape{0, 0, 0, 0};
  wshape[N] = param.num_filter;
  wshape[H] = param.kernel[0];
  wshape[W] = param.kernel[1];
  wshape[C] = dshape[C];
  SHAPE_ASSIGN_CHECK(*in_shape, 1, wshape);
  const int start = param.no_bias? 2 : 3;
  const int end = param.no_bias? 6 : 9;
  for (int i = start; i < end; ++i) {
    SHAPE_ASSIGN_CHECK(*in_shape, i, TShape{1});
  }
  if (!param.no_bias) {
    SHAPE_ASSIGN_CHECK(*in_shape, 2, Shape1(param.num_filter));
  }

  auto AddPad = [](index_t dsize, index_t pad) { return dsize + 2 * pad; };
  TShape oshape{1, 1, 1, 1};
  oshape[N] = dshape[N];
  oshape[C] = wshape[N];
  oshape[H] = (AddPad(dshape[H], param.pad[0]) - wshape[H]) / param.stride[0] + 1;
  oshape[W] = (AddPad(dshape[W], param.pad[1]) - wshape[W]) / param.stride[1] + 1;

  out_shape->clear();
  out_shape->push_back(oshape);
  out_shape->push_back(TShape{1});
  out_shape->push_back(TShape{1});
  return true;
}

inline bool QuantizedConvType(const nnvm::NodeAttrs& attrs,
                              std::vector<int> *in_type,
                              std::vector<int> *out_type) {
  const QuantizedConvParam& param = nnvm::get<QuantizedConvParam>(attrs.parsed);
  CHECK_EQ(in_type->size(), param.no_bias? 6U : 9U);
  TYPE_ASSIGN_CHECK(*in_type, 0, mshadow::kInt8);
  TYPE_ASSIGN_CHECK(*in_type, 1, mshadow::kInt8);
  if (!param.no_bias) {
    TYPE_ASSIGN_CHECK(*in_type, 2, mshadow::kInt8);
  }

  const size_t start = param.no_bias? 2 : 3;
  const size_t end = param.no_bias? 6 : 9;
  for (size_t i = start; i < end; ++i) {
    TYPE_ASSIGN_CHECK(*in_type, i, mshadow::kFloat32);
  }

  out_type->clear();
  out_type->push_back(mshadow::kInt32);
  out_type->push_back(mshadow::kFloat32);
  out_type->push_back(mshadow::kFloat32);
  return true;
}

template<typename xpu>
void QuantizedConvForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs);

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_QUANTIZED_CONV_H_
