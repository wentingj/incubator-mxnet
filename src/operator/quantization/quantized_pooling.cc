/*!
 * Copyright (c) 2017 by Contributors
 * \file quantized_pooling.cc
*/
#include <mxnet/op_attr_types.h>
#include "./quantized_pooling-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(QuantizedPoolingParam param, int dtype) {
  Operator *op = NULL;
  LOG(FATAL) << "not implemented";
  // MSHADOW_TYPE_SWITCH(dtype, DType, {
  //   op = new QuantizedMaxPoolOp<cpu, DType>(param);
  // });
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator* QuantizedPoolingProp::CreateOperatorEx(Context ctx,
  std::vector<TShape> *in_shape, std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(QuantizedPoolingParam);

MXNET_REGISTER_OP_PROPERTY(quantized_pooling, QuantizedPoolingProp)
.describe(R"code(Performs pooling on the input.

The shapes for 1-D pooling are

- **data**: *(batch_size, channel, width)*,
- **out**: *(batch_size, num_filter, out_width)*.

The shapes for 2-D pooling are

- **data**: *(batch_size, channel, height, width)*
- **out**: *(batch_size, num_filter, out_height, out_width)*, with::

    out_height = f(height, kernel[0], pad[0], stride[0])
    out_width = f(width, kernel[1], pad[1], stride[1])

The defintion of *f* depends on ``pooling_convention``, which has two options:

- **valid** (default)::

    f(x, k, p, s) = floor(x+2*p-k)/s+1

- **full**, which is compatible with Caffe::

    f(x, k, p, s) = ceil(x+2*p-k)/s+1

But ``global_pool`` is set to be true, then do a global pooling, namely reset
``kernel=(height, width)``.

For 3-D pooling, an additional *depth* dimension is added before
*height*. Namely the input data will have shape *(batch_size, channel, depth,
height, width)*.

)code" ADD_FILELINE)
#if 0
.set_attr<FNeedRequantize>("FNeedRequantize",
  [](const NodeAttrs& attrs) {
    const QuantizedPoolingParam& param = nnvm::get<QuantizedPoolingParam>(attrs.parsed);
    if (param.pool_type == pool_enum::kMaxPooling || param.pool_type == pool_enum::kAvgPooling) {
      return false;
    } else {
      return true;
    }
  })
#endif
.add_argument("data", "NDArray-or-Symbol", "Input data to the pooling operator.")
.add_argument("min_range", "NDArray-or-Symbol", "")
.add_argument("max_range", "NDArray-or-Symbol", "")
.add_arguments(QuantizedPoolingParam::__FIELDS__());


NNVM_REGISTER_OP(Pooling)
.set_attr<FQuantizedOp>("FQuantizedOp", [](nnvm::NodePtr n) {
    const NodeAttrs& attrs = n->attrs;
    QuantizedPoolingParam param;
    param.Init(attrs.dict);
    nnvm::NodePtr node = nnvm::Node::Create();
    if (param.pool_type == pool_enum::kMaxPooling || param.pool_type == pool_enum::kAvgPooling) {
      node->attrs.op = Op::Get("quantized_pooling");
      node->attrs.name = "quantized_" + attrs.name;
    } else {
      node->attrs.op = Op::Get("Pooling");
      node->attrs.name = attrs.name;
    }
    node->attrs.dict = attrs.dict;
    if (node->op()->attr_parser != nullptr) {
      node->op()->attr_parser(&(node->attrs));
    }
    return node;
  });

}  // namespace op
}  // namespace mxnet
