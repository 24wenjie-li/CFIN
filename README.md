# CFIN

This repository is an official PyTorch implementation of our paper "Cross-receptive Focused Inference Network for Lightweight Image Super-Resolution". 

## Prerequisites:
```
1. Python >= 3.6
2. PyTorch >= 1.2
3. numpy
4. skimage
5. imageio
6. tqdm
7. timm
8. einops
```

## Dataset
We used only DIV2K dataset to train our model. To speed up the data reading during training, we converted the format of the images within the dataset from png to npy. Please download the DIV2K_decoded with npy format from <a href="https://pan.quark.cn/s/43248032bab2">here</a>.[Quark Netdisk][Password:None]

The test set contains five datasets, Set5, Set14, B100, Urban100, Manga109. The benchmark can be downloaded from <a href="https://pan.baidu.com/s/1Vb68GWERriLmJRtYfm2uEg">here</a>.[Baidu Netdisk][Password:8888]

Extract the file and place it in the same location as args.data_dir in option.py.

```
├── CFIN  					  # Code
├── dataset  					# all datasets for this code
|  └── DIV2K_decoded 				#  train datasets
|  |  └── DIV2K_train_HR  		
|  |  └── DIV2K_train_LR_bicubic 			
|  └── benchmark 					  #  test datasets 
|  |  └── Set5
|  |  └── Set14
|  |  └── B100
|  |  └── Urban100
|  |  └── Manga109
```


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
