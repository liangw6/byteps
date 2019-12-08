#!/bin/bash

path="`dirname $0`"

if [ "$EVAL_TYPE" == "mnist" ]; then
    echo "training mnist..."
    python $path/train_mnist_byteps.py $@
elif [ "$EVAL_TYPE" == "imagenet" ]; then
    echo "training imagenet..."
    python $path/train_imagenet_resnet50_byteps.py $@
elif [ "$EVAL_TYPE" == "benchmark" ]; then
    echo "running benchmark..."
    python $path/benchmark_byteps.py $@
elif [ "$EVAL_TYPE" == "microbenchmark" ]; then
    echo "running microbenchmark"
    python $path/microbenchmark-byteps.py $@
if [ "$EVAL_TYPE" == "mnist_inference" ]; then
    echo "training mnist... with real-time inference"
    python $path/mnist_client.py $@
else
  echo "Error: unsupported $EVAL_TYPE"
  exit 1
fi
