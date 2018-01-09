#! /bin/sh

python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-symbol.json --param-file=./model/imagenet1k-resnet-152-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec

python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec

python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-10batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-50batches-naive-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec

python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-5batches-entropy-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-10batches-entropy-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
python imagenet_inference.py --symbol-file=./model/imagenet1k-resnet-152-quantized-50batches-entropy-symbol.json --param-file=./model/imagenet1k-resnet-152-quantized-0000.params --rgb-mean=0,0,0 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec


python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-symbol.json --param-file=./model/imagenet1k-inception-bn-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec

python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec

python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-5batches-naive-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-10batches-naive-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-50batches-naive-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec

python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-5batches-entropy-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-10batches-entropy-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
python imagenet_inference.py --symbol-file=./model/imagenet1k-inception-bn-quantized-50batches-entropy-symbol.json --param-file=./model/imagenet1k-inception-bn-quantized-0000.params --rgb-mean=123.68,116.779,103.939 --num-skipped-batches=50 --num-inference-batches=500 --dataset=./data/val_256_q90.rec
