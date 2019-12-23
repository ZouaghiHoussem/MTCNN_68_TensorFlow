#!/bin/bash
set -e
### All of your tmp data will be saved in ./tmp folder

echo "Hello! I will prepare training data and starting to training step by step."

echo "Start to training R-Net"
python training/train.py --stage=rnet


