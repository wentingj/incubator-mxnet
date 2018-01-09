#! /bin/sh

python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-method=none

python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=5 --calib-method=naive
python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=10 --calib-method=naive
python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=50 --calib-method=naive

python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=5 --calib-method=entropy
python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=10 --calib-method=entropy
python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=50 --calib-method=entropy


python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-method=none

python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=5 --calib-method=naive
python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=10 --calib-method=naive
python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=50 --calib-method=naive

python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=5 --calib-method=entropy
python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=10 --calib-method=entropy
python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=50 --calib-method=entropy
