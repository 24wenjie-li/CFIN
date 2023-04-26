# CFIN

This repository is an official PyTorch implementation of our paper "Cross-receptive Focused Inference Network for Lightweight Image Super-Resolution". Accepted by IEEE TRANSACTIONS ON MULTIMEDIA, 2023.

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
We used only DIV2K dataset to train our model. To speed up the data reading during training, we converted the format of the images within the dataset from png to npy. 
Please download the DIV2K_decoded with npy format from <a href="https://pan.quark.cn/s/43248032bab2">here</a>.[Quark Netdisk]

The test set contains five datasets, Set5, Set14, B100, Urban100, Manga109. The benchmark can be downloaded from <a href="https://pan.baidu.com/s/1Vb68GWERriLmJRtYfm2uEg">here</a>.[Baidu Netdisk][Password:8888]

Extract the file and place it in the same location as args.data_dir in option.py.

The code and datasets need satisfy the following structures:
```
├── CFIN  					# Train / Test Code
├── dataset  					# all datasets for this code
|  └── DIV2K_decoded  		#  train datasets with npy format
|  |  └── DIV2K_train_HR  		
|  |  └── DIV2K_train_LR_bicubic 			
|  └── benchmark  		#  test datasets with png format 
|  |  └── Set5
|  |  └── Set14
|  |  └── B100
|  |  └── Urban100
|  |  └── Manga109
 ─────────────────
```


## Results
All our SR Results can be downloaded from <a href="https://pan.baidu.com/s/1QVku7exoRGRNNwKeWUThAw">here</a>.[Baidu Netdisk][Password:8888]

All pretrained model can be found in .

We will release this code soon!

## Training
Note：
```
  # CFIN x2
  python main.py --scale 2 --model CFINx2 --patch_size 96 --save experiments/CFINx2
  
  # CFIN x3
  python main.py --scale 3 --model CFINx3 --patch_size 144 --save experiments/CFINx3
  
  # CFIN x4
  python main.py --scale 4 --model CFINx4 --patch_size 192 --save experiments/CFINx4
```

## Testing
Since the PSNR/SSIM values in our paper are obtained from the Matlab program, the data obtained using the python code may have a floating error of 0.01 dB in the PSNR. The following PSNR/SSIMs are evaluated on Matlab R2017a and the code can be referred to <a href="https://github.com/24wenjie-li/FDIWN/blob/main/FDIWN_TestCode/Evaluate_PSNR_SSIM.m">here</a>.(You need to modify the test path!)
```
# CFIN x2
python main.py --scale 2 --model CFINx2 --save test_results/CFINx2 --pre_train experiments/CFIN/model/model_best_x2.pt --test_only --save_results --data_test Set5

# CFIN x3
python main.py --scale 3 --model CFINx3 --save test_results/CFINx3 --pre_train experiments/CFIN/model/model_best_x3.pt --test_only --save_results --data_test Set5

# CFIN x4
python main.py --scale 4 --model CFINx4 --save test_results/CFINx4 --pre_train experiments/CFIN/model/model_best_x4.pt --test_only --save_results --data_test Set5

# CFIN+ x2 with self-ensemble strategy
python main.py --scale 2 --model CFINx2 --save test_results/CFINx2 --pre_train experiments/CFIN/model/model_best_x2.pt --test_only --save_results --chop --self_ensemble --data_test Set5 

# CFIN+ x3 with self-ensemble strategy
python main.py --scale 3 --model CFINx3 --save test_results/CFINx3 --pre_train experiments/CFIN/model/model_best_x3.pt --test_only --save_results --chop --self_ensemble --data_test Set5

# CFIN+ x4 with self-ensemble strategy
python main.py --scale 4 --model CFINx4 --save test_results/CFINx4 --pre_train experiments/CFIN/model/model_best_x4.pt --test_only --save_results --chop --self_ensemble --data_test Set5
```

## Test Parmas and Mutii-adds
Note：You need to install torchsummaryX!
```
# Default test CFINx4
python test_summary.py
```

## Performance
Our CFIN is trained on RGB, but as in previous work, we only reported PSNR/SSIM on the Y channel.

Model|Scale|Params|Multi-adds|Set5|Set14|B100|Urban100|Manga109
--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:
CFIN        |x2|675K|116.9G|38.14/0.9610|33.80/0.9199|32.26/0.9006|32.48/0.9311|38.97/0.9777
CFIN        |x3|681K|53.5G|34.65/0.9289|30.45/0.8443|29.18/0.8071|28.49/0.8583|33.89/0.9464
CFIN        |x4|699K|31.2G|32.49/0.8985|28.74/0.7849|27.68/0.7396|26.39/0.7946|30.73/0.9124

## Some extra questions
You can download the supplementary materials on issues requested by reviewers from <a href="">here</a>.

## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [DRN](https://github.com/guoyongcs/DRN). We thank the authors for sharing their codes.

## Citation

If you use any part of this code in your research, please cite our paper:

```

```
