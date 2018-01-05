#! /bin/sh
MXNET_ENGINE_TYPE=NaiveEngine python "$1" --gpus=0 --data-nthreads=60 --data-val=./data/val_256_q90.rec