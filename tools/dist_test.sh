#!/usr/bin/env bash
# UniUncer: Switched from deprecated torch.distributed.launch to torch.distributed.run for PyTorch 2.0+.

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29611}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
