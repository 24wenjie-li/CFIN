# CFIN

This repository is an official PyTorch implementation of our paper "Cross-receptive Focused Inference Network for Lightweight Image Super-Resolution". 

## Prerequisites:
1. Python >= 3.6
2. PyTorch >= 1.2
3. numpy
4. skimage
5. imageio
6. matplotlib
7. tqdm
8. timm
9. einops

## Dataset
We used only DIV2K dataset to train our model. To speed up the data reading during training, we converted the format of the images within the dataset from png to npy. Please download the DIV2K_decoded with npy format from <a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">here</a>.[Baidu Netdisk][Password:8888]

The test set contains five datasets, Set5, Set14, B100, Urban100, Manga109. They can be downloaded from <a href="https://pan.baidu.com/s/1XwdEjCgiPfHTumGU4aWKiQ">here</a>.[Baidu Netdisk][Password:8888]

Extract the file and place it in the same location as args.data_dir in option.py.

## Results
All our SR Results can be downloaded from <a href="https://pan.baidu.com/s/1QVku7exoRGRNNwKeWUThAw">here</a>.[Baidu Netdisk][Password:8888]

All pretrained model can be found in .

We will release this code soon!

## Training
```
  # CFIN x2  LR: 48 * 48  HR: 96 * 96
  python main.py
  
  # CFIN x3  LR: 48 * 48  HR: 144 * 144
  python main.py
  
  # CFIN x4  LR: 48 * 48  HR: 192 * 192
  python main.py
```

## Testing
