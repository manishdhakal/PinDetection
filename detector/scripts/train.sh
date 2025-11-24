#!/bin/bash

python detector/train.py \
    --data-root detector/data \
    --n-folds 5 \
    --input-size 12 \
    --num-classes 10 \
    --batch-size 256 \
    --lr 0.01 \
    --lr-min 0.00001 \
    --num-epochs 1000 \
    --seed 42 \
    --log-path detector/logs \
    --device cuda:0