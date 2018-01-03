/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_conv.cc
 * \brief
 * \author Ziheng Jiang, Jun Wu
*/
#include "./quantized_conv-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_quantized_conv)
.set_num_inputs(
  [](const NodeAttrs& attrs) {
    const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
    return param.no_bias? 6 : 9;
  })
.set_num_outputs(3)
.set_attr_parser(ParamParser<ConvolutionParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
    if (param.no_bias) {
      return std::vector<std::string>{"data", "weight", "min_data", "max_data",
                                      "min_weight", "max_weight"};
    } else {
      return std::vector<std::string>{"data", "weight", "bias", "min_data", "max_data",
                                      "min_weight", "max_weight", "min_bias", "max_bias"};
    }
  })
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output", "min_output", "max_output"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", QuantizedConvShape)
.set_attr<nnvm::FInferType>("FInferType", QuantizedConvType)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>(1, ResourceRequest::kTempSpace);
  })
.set_attr<TQuantizationNeedShrink>("TQuantizationNeedShrink", true)
.add_argument("data", "NDArray-or-Symbol", "Input data.")
.add_argument("weight", "NDArray-or-Symbol", "weight.")
.add_argument("bias", "NDArray-or-Symbol", "bias.")
.add_argument("min_data", "NDArray-or-Symbol", "Minimum value of data.")
.add_argument("max_data", "NDArray-or-Symbol", "Maximum value of data.")
.add_argument("min_weight", "NDArray-or-Symbol", "Minimum value of weight.")
.add_argument("max_weight", "NDArray-or-Symbol", "Maximum value of weight.")
.add_argument("min_bias", "NDArray-or-Symbol", "Minimum value of bias.")
.add_argument("max_bias", "NDArray-or-Symbol", "Maximum value of bias.")
.add_arguments(ConvolutionParam::__FIELDS__());

NNVM_REGISTER_OP(Convolution)
.set_attr<FQuantizedOp>("FQuantizedOp", [](nnvm::NodePtr n) {
    const nnvm::NodeAttrs& attrs = n->attrs;
    nnvm::NodePtr node = nnvm::Node::Create();
    node->attrs.op = Op::Get("_contrib_quantized_conv");
    node->attrs.name = "quantized_" + attrs.name;
    node->attrs.dict = attrs.dict;
    if (node->op()->attr_parser != nullptr) {
      node->op()->attr_parser(&(node->attrs));
    }
    return node;
  });

}  // namespace op
}  // namespace mxnet
