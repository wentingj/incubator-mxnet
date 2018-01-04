/*!
 *  Copyright (c) 2017 by Contributors
 * \file quantize.cu
 * \brief
 */
#include "./requantize-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_requantize)
.set_attr<FCompute>("FCompute<gpu>", RequantizeForward<gpu>);

}  // namespace op
}  // namespace mxnet
