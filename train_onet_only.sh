#!/bin/bash
set -e
### All of your tmp data will be saved in ./tmp folder

echo "Hello! I will prepare training data and starting to training step by step."


# 4. stage: O-Net
### generate training data(Face Detection Part) for ONet
echo "Preparing O-Net training data: bbox"
python2 prepare_data/gen_hard_bbox_rnet_onet.py --stage=onet
### generate training data(Face Landmark Detection Part) for ONet
echo "Preparing O-Net training data: landmark"
python2 prepare_data/gen_landmark_aug.py --stage=onet
### generate tfrecord file for tf training
echo "Preparing O-Net tfrecord file"
python2 prepare_data/gen_tfrecords.py --stage=onet
### start to training O-Net
echo "Start to training O-Net"
python2 training/train.py --stage=onet

# 5. Done
echo "Congratulation! All stages had been done. Now you can going to testing and hope you enjoy your result."
echo "haha...bye bye"

