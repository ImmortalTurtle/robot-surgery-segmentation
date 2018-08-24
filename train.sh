#!/bin/bash

model=UNet11

python3 train.py --batch-size 12 --root models --workers 4 --lr 0.0001 --n-epochs 35 --jaccard-weight 0.3 --model $model
