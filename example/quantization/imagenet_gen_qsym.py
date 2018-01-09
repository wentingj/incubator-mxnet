import argparse
from common import modelzoo
import mxnet as mx
import os
import logging
from mxnet.quantization import *


def download_calib_dataset(dataset_url, calib_dataset, logger=None):
    if logger is not None:
        logger.info('Downloading calibration dataset from %s to %s' % (dataset_url, calib_dataset))
    mx.test_utils.download(dataset_url, calib_dataset)


def download_model(model_name, logger=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, 'model')
    if logger is not None:
        logger.info('Downloading model %s... into path %s' % (model_name, model_path))
    return modelzoo.download_model(args.model, os.path.join(dir_path, 'model'))


def save_symbol(fname, sym, logger=None):
    if logger is not None:
        logger.info('Saving symbol into file at %s' % fname)
    sym.save(fname)


def save_params(fname, arg_params, aux_params, logger=None):
    if logger is not None:
        logger.info('Saving params into file at %s' % fname)
    save_dict = {('arg:%s' % k) : v.as_in_context(cpu()) for k, v in arg_params.items()}
    save_dict.update({('aux:%s' % k) : v.as_in_context(cpu()) for k, v in aux_params.items()})
    mx.nd.save(fname, save_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a calibrated quantized model from a FP32 model')
    parser.add_argument('--model', type=str, required=True,
                        help='currently only supports imagenet1k-resnet-152 or imagenet1k-inception-bn')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--label-name', type=str, default='softmax_label')
    parser.add_argument('--calib-dataset', type=str, default='data/val_256_q90.rec',
                        help='path of the calibration dataset')
    parser.add_argument('--image-shape', type=str, default='3,224,224')
    parser.add_argument('--data-nthreads', type=int, default=60,
                        help='number of threads for data decoding')
    parser.add_argument('--num-calib-batches', type=int, default=10,
                        help='number of batches for calibration')
    parser.add_argument('--exclude-first-conv', action='store_true', default=True,
                        help='excluding quantizing the first conv layer since the'
                             ' number of channels is usually not a multiple of 4 in that layer'
                             ' which does not satisfy the requirement of cuDNN')
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
    parser.add_argument('--calib-method', type=str, default='entropy',
                        help='calibration method used for generating calibration table for the quantized symbol; supports'
                             ' 1. none: no calibration method will be used. The thresholds for quantization will be'
                             ' calculated on the fly. This will result in runtime penalty and loss of accuracy in general.'
                             ' 2. naive: simply take min and max values of layer outputs as thresholds for quantization'
                             ' 3. entropy: calculate KL divergence of the fp32 output and quantized output for optimal'
                             ' thresholds. This method takes much more time than the naive method, but results more'
                             ' accurate inference results.')
    args = parser.parse_args()

    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    logger.info('shuffle_dataset=%s' % args.shuffle_dataset)

    calib_method = args.calib_method
    logger.info('calibration method set to %s' % calib_method)

    # download calibration dataset
    if calib_method != 'none':
        download_calib_dataset('http://data.mxnet.io/data/val_256_q90.rec', args.calib_dataset)

    # download model
    prefix, epoch = download_model(model_name=args.model, logger=logger)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # get batch size
    batch_size = args.batch_size
    logger.info('batch size = %d for calibration' % batch_size)

    # get number of batches for calibration
    num_calib_batches = args.num_calib_batches
    if calib_method != 'none':
        logger.info('number of batches = %d for calibration' % num_calib_batches)

    # get number of threads for decoding the dataset
    data_nthreads = args.data_nthreads

    # get image shape
    image_shape = args.image_shape

    exclude_first_conv = args.exclude_first_conv
    excluded_sym_names = []
    if args.model == 'imagenet1k-resnet-152':
        rgb_mean = '0,0,0'
        include_layer = lambda name: name.endswith('_output') and (name.find('conv') != -1
                                                                   or name.find('sc') != -1
                                                                   or name.find('fc') != -1)
        if exclude_first_conv:
            excluded_sym_names = ['conv0']
    elif args.model == 'imagenet1k-inception-bn':
        rgb_mean = '123.68,116.779,103.939'
        include_layer = lambda name: name.endswith('_output') and (name.find('conv') != -1
                                                                   or name.find('fc') != -1)
        if exclude_first_conv:
            excluded_sym_names = ['conv_1']
    else:
        raise ValueError('model %s is not supported in this script' % args.model)

    label_name = args.label_name
    logger.info('label_name = %s' % label_name)

    data_shape = tuple([int(i) for i in image_shape.split(',')])
    logger.info('Input data shape = %s' % str(data_shape))

    logger.info('rgb_mean = %s' % rgb_mean)
    rgb_mean = [float(i) for i in rgb_mean.split(',')]
    mean_args = {'mean_r': rgb_mean[0], 'mean_g': rgb_mean[1], 'mean_b': rgb_mean[2]}


    # cudnn int8 convolution only supports #channels as a multiple of 4
    # have to skip quantizing the first conv layer
    excluded_symbols = []
    for name in excluded_sym_names:
        nodes = sym.get_internals()
        idx = nodes.list_outputs().index(name + '_output')
        excluded_symbols.append(nodes[idx])

    logger.info('Quantizing FP32 model %s' % args.model)
    qsym = quantize_symbol(sym, excluded_symbols=excluded_symbols, offline_params=arg_params.keys())
    sym_name = '%s-symbol.json' % (prefix + '-quantized')
    save_symbol(sym_name, qsym, logger)

    logger.info('Quantizing parameters FP32 model %s' % args.model)
    qarg_params = quantize_params(qsym, arg_params)
    param_name = '%s-%04d.params' % (prefix + '-quantized', epoch)
    save_params(param_name, qarg_params, aux_params, logger)

    if calib_method != 'none':
        logger.info('Creating ImageRecordIter for reading calibration dataset')
        data = mx.io.ImageRecordIter(path_imgrec=args.calib_dataset,
                                     label_width=1,
                                     preprocess_threads=data_nthreads,
                                     batch_size=batch_size,
                                     data_shape=data_shape,
                                     label_name=label_name,
                                     rand_crop=False,
                                     rand_mirror=False,
                                     shuffle=args.shuffle_dataset,
                                     shuffle_chunk_seed=args.shuffle_chunk_seed,
                                     seed=args.shuffle_seed,
                                     **mean_args)

        mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), label_names=[label_name, ])
        mod.bind(for_training=False, data_shapes=data.provide_data, label_shapes=data.provide_label)
        mod.set_params(arg_params, aux_params)
        num_calib_images = num_calib_batches * batch_size

        if calib_method == 'entropy':
            logger.info('Collecting layer outputs from FP32 model using %d batches' % num_calib_batches)
            nd_dict = collect_layer_outputs(mod, data, include_layer=include_layer, max_num_examples=num_calib_images,
                                            logger=logger)
            logger.info('Calculating optimal thresholds for quantization')
            th_dict = mx.quantization.get_optimal_thresholds(nd_dict, logger=logger)
            suffix = '-quantized-%dbatches-entropy' % num_calib_batches
        elif calib_method == 'naive':
            logger.info('Collecting layer output min/max values from FP32 model using %d batches' % num_calib_batches)
            th_dict = mx.quantization.collect_layer_output_min_max(mod, data, include_layer=include_layer,
                                                                   max_num_examples=num_calib_images,
                                                                   logger=logger)
            suffix = '-quantized-%dbatches-naive' % num_calib_batches
        else:
            raise ValueError('unknow calibration method %s entered' % calib_method)

        logger.info('Calibrating quantized model...')
        cqsym = mx.quantization.calibrate_quantized_sym(qsym, th_dict)
        sym_name = '%s-symbol.json' % (prefix + suffix)
        save_symbol(sym_name, cqsym, logger)
