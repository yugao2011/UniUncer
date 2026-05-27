#!/usr/bin/env bash
# UniUncer: Switched from deprecated torch.distributed.launch to torch.distributed.run for PyTorch 2.0+.

CONFIG=$1
GPUS=$2
PORT=${PORT:-28651}
# CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
