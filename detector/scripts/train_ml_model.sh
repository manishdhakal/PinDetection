#!/bin/bash


uv run detector/svm_train.py \
    --data-file-name detector/data/combined_dataset.csv \
    --combined True \
    --n-folds 5 \
    --input-size 12 \
    --num-classes 10 \
    --seed 42 \
    --result-path detector/results/ml_model/svm/combined/ \

uv run detector/svm_train.py \
    --data-file-name detector/data/test_dataset.csv \
    --combined True \
    --n-folds 5 \
    --input-size 12 \
    --num-classes 10 \
    --seed 42 \
    --result-path detector/results/ml_model/svm/test_dataset/ \

uv run detector/svm_train.py \
    --data-file-name detector/data/anjila_dataset.csv \
    --combined False \
    --n-folds 5 \
    --input-size 12 \
    --num-classes 10 \
    --seed 42 \
    --result-path detector/results/ml_model/svm/individual-anjila/ \

uv run detector/svm_train.py \
    --data-file-name detector/data/dataset.txt \
    --combined False \
    --n-folds 5 \
    --input-size 12 \
    --num-classes 10 \
    --seed 42 \
    --result-path detector/results/ml_model/svm/individual-manish/ \



# # FOR SVM

# uv run detector/dt_train.py \
#     --data-file-name detector/data/combined_dataset.csv \
#     --combined True \
#     --n-folds 5 \
#     --input-size 12 \
#     --num-classes 10 \
#     --seed 42 \
#     --result-path detector/results/ml_model/dt/combined/ \
#     --max-depth-dt 5 \

# uv run detector/dt_train.py \
#     --data-file-name detector/data/test_dataset.csv \
#     --combined True \
#     --n-folds 5 \
#     --input-size 12 \
#     --num-classes 10 \
#     --seed 42 \
#     --result-path detector/results/ml_model/dt/test_dataset/ \
#     --max-depth-dt 5 \

# uv run detector/dt_train.py \
#     --data-file-name detector/data/anjila_dataset.csv \
#     --combined False \
#     --n-folds 5 \
#     --input-size 12 \
#     --num-classes 10 \
#     --seed 42 \
#     --result-path detector/results/ml_model/dt/individual-anjila/ \
#     --max-depth-dt 5 \

# uv run detector/dt_train.py \
#     --data-file-name detector/data/dataset.txt \
#     --combined False \
#     --n-folds 5 \
#     --input-size 12 \
#     --num-classes 10 \
#     --seed 42 \
#     --result-path detector/results/ml_model/dt/individual-manish/ \
#     --max-depth-dt 5 \