# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import argparse
import mxnet as mx
import time
import os
import logging
from mxnet.quantization import *
from sklearn.datasets import fetch_mldata


def download_dataset(dataset_url, dataset_dir, logger=None):
    if logger is not None:
        logger.info('Downloading dataset for inference from %s to %s' % (dataset_url, dataset_dir))
    mx.test_utils.download(dataset_url, dataset_dir)


def load_model(symbol_file, param_file, logger=None):
    cur_path = os.path.dirname(os.path.realpath(__file__))
    symbol_file_path = os.path.join(cur_path, symbol_file)
    if logger is not None:
        logger.info('Loading symbol from file %s' % symbol_file_path)
    symbol = mx.sym.load(symbol_file_path)

    param_file_path = os.path.join(cur_path, param_file)
    if logger is not None:
        logger.info('Loading params from file %s' % param_file_path)
    save_dict = nd.load(param_file_path)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return symbol, arg_params, aux_params


def advance_data_iter(data_iter, n):
    assert n >= 0
    if n == 0:
        return data_iter
    has_next_batch = True
    while has_next_batch:
        try:
            data_iter.next()
            n -= 1
            if n == 0:
                return data_iter
        except StopIteration:
            has_next_batch = False

def score_conv_mnist(json_file, sym):
    # prepare data
    mnist = fetch_mldata('MNIST original')
    np.random.seed(1234) # set seed for deterministic ordering
    p = np.random.permutation(mnist.data.shape[0])
    X = mnist.data[p].reshape(70000, 1, 28, 28)
    #X = mnist.data[p].reshape(70000, 28, 28, 1)
    #X = mnist.data[p].reshape(70000, 4, 7, 28)
    pad = np.zeros(shape=(70000, 15, 28, 28))
    #pad = np.zeros(shape=(70000, 28, 28, 3))
    #pad = np.zeros(shape=(70000, 12, 7, 28))
    X = np.concatenate([X, pad], axis=1)
    #X = np.concatenate([X, pad], axis=3)
    #X = np.concatenate([X, pad], axis=1)
    Y = mnist.target[p]
    
    X = X.astype(np.uint8)/255
    X_train = X[:60000]
    X_test = X[60000:]
    Y_train = Y[:60000]
    Y_test = Y[60000:]
    
    train_iter = mx.io.NDArrayIter(X_train, Y_train, batch_size=batch_size)
    val_iter = mx.io.NDArrayIter(X_test, Y_test, batch_size=batch_size)
    test_iter = val_iter
    # create a trainable module on GPU 0
    model = mx.mod.Module(symbol=sym, context=mx.cpu(0))
   
    if (args.symbol_file == 'conv_mnist-symbol.json') :
        _, arg_params, aux_params = mx.model.load_checkpoint("conv_mnist", 10)
    else :
        _, arg_params, aux_params = mx.model.load_checkpoint("conv_mnist-quantized", 10)
    #print('arg_params=', arg_params)
    model.bind(for_training=False, data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    print('after bind')
    model.set_params(arg_params=arg_params, aux_params=aux_params)
    print('after set_param')
    
    # predict accuracy for conv net
    acc = mx.metric.Accuracy()
    print('Accuracy: {}%'.format(model.score(test_iter, acc)[0][1]*100))
    
    #print(quantized_conv_net.debug_str())
    params = model.get_params()[0]
    # print(params['convolution0_weight'].asnumpy())
    
    print('before test')
    test(sym, params, test_iter)
    print('after test')


def test(symbol, params, test_iter):
    model = mx.model.FeedForward(
        symbol,
        ctx=mx.cpu(0),
        arg_params=params)
    print('Accuracy:', model.score(test_iter)*100, '%')


def score(sym, arg_params, aux_params, data, devs, label_name, max_num_examples, logger=None):
    metrics = [mx.metric.create('acc'),
               mx.metric.create('top_k_accuracy', top_k=5)]
    if not isinstance(metrics, list):
        metrics = [metrics, ]
    mod = mx.mod.Module(symbol=sym, context=devs, label_names=[label_name, ])
    mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
    mod.set_params(arg_params, aux_params)
    

    tic = time.time()
    num = 0
    for batch in data:
        mod.forward(batch, is_train=False)
        for m in metrics:
            mod.update_metric(m, batch.label)
        num += batch_size
        if max_num_examples is not None and num >= max_num_examples:
            break

    speed = num / (time.time() - tic)

    if logger is not None:
        logger.info('Finished inference with %d images' % num)
        logger.info('Finished with %f images per second', speed)
        for m in metrics:
            logger.info(m.get())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score a model on a dataset')
    parser.add_argument('--symbol-file', type=str, required=True, help='symbol file path')
    parser.add_argument('--param-file', type=str, required=True, help='param file path')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--label-name', type=str, default='softmax_label')
    parser.add_argument('--dataset', type=str, required=True, help='dataset path')
    parser.add_argument('--rgb-mean', type=str, default='0,0,0')
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--data-nthreads', type=int, default=60, help='number of threads for data decoding')
    parser.add_argument('--num-skipped-batches', type=int, default=0, help='skip the number of batches for inference')
    parser.add_argument('--num-inference-batches', type=int, required=True, help='number of images used for inference')
    parser.add_argument('--shuffle-dataset', action='store_true', default=True,
                        help='shuffle the calibration dataset')
    parser.add_argument('--shuffle-chunk-seed', type=int, default=3982304,
                        help='shuffling chunk seed, see'
                             ' https://mxnet.incubator.apache.org/api/python/io/io.html?highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')
    parser.add_argument('--shuffle-seed', type=int, default=48564309,
                        help='shuffling seed, see'
                             ' https://mxnet.incubator.apache.org/api/python/io/io.html?highlight=imager#mxnet.io.ImageRecordIter'
                             ' for more details')

    args = parser.parse_args()

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    symbol_file = args.symbol_file
    param_file = args.param_file
    data_nthreads = args.data_nthreads

    batch_size = args.batch_size
    logger.info('batch size = %d for inference' % batch_size)

    rgb_mean = args.rgb_mean
    logger.info('rgb_mean = %s' % rgb_mean)
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    mean_args = {'mean_r': rgb_mean[0], 'mean_g': rgb_mean[1], 'mean_b': rgb_mean[2]}

    if ((args.symbol_file != 'conv_mnist-symbol.json') & (args.symbol_file != 'conv_mnist-quantized-symbol.json')) :
        label_name = args.label_name
        logger.info('label_name = %s' % label_name)

        image_shape = args.image_shape
        data_shape = tuple([int(i) for i in image_shape.split(',')])
        logger.info('Input data shape = %s' % str(data_shape))

        dataset = args.dataset
        download_dataset('http://data.mxnet.io/data/val_256_q90.rec', dataset)
        logger.info('Dataset for inference: %s' % dataset)

        # creating data iterator
        data = mx.io.ImageRecordIter(path_imgrec=dataset,
                                     label_width=1,
                                     preprocess_threads=data_nthreads,
                                     batch_size=batch_size,
                                     data_shape=data_shape,
                                     label_name=label_name,
                                     rand_crop=False,
                                     rand_mirror=False,
                                     shuffle=True,
                                     shuffle_chunk_seed=3982304,
                                     seed=48564309,
                                     **mean_args)

        # loading model
        sym, arg_params, aux_params = load_model(symbol_file, param_file, logger)

        # make sure that fp32 inference works on the same images as calibrated quantized model
        logger.info('Skipping the first %d batches' % args.num_skipped_batches)
        data = advance_data_iter(data, args.num_skipped_batches)

        num_inference_images = args.num_inference_batches * batch_size
        logger.info('Running model %s for inference' % symbol_file)
        score(sym, arg_params, aux_params, data, [mx.cpu()], label_name,
              max_num_examples=num_inference_images, logger=logger)
    else:
       # loading model
       sym, arg_params, aux_params = load_model(symbol_file, param_file, logger)
       score_conv_mnist(args.symbol_file, sym)
