#! /bin/sh

python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-mode=none

python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=5 --calib-mode=naive
python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=10 --calib-mode=naive
python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=50 --calib-mode=naive

python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=5 --calib-mode=entropy
python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=10 --calib-mode=entropy
python imagenet_gen_qsym.py --model=imagenet1k-resnet-152 --calib-dataset=./data/val_256_q90.rec --num-calib-batches=50 --calib-mode=entropy


python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-mode=none

python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=5 --calib-mode=naive
python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=10 --calib-mode=naive
python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=50 --calib-mode=naive

python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=5 --calib-mode=entropy
python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=10 --calib-mode=entropy
python imagenet_gen_qsym.py --model=imagenet1k-inception-bn --calib-dataset=./data/val_256_q90.rec --num-calib-batches=50 --calib-mode=entropy
