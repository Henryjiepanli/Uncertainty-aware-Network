# Codes for 《 UANet: an Uncertainty-Aware Network for Building Extraction from Remote Sensing Images》

## About Paper
We are delighted to inform everyone that our paper has been successfully accepted by TGRS (IEEE Transactions on Geoscience and Remote Sensing). 
[Paper Link](https://ieeexplore.ieee.org/document/10418227)

The results on the three building datasets can be downloaded via Baidu Disk：[Link](https://pan.baidu.com/s/1MkoWfIyz7DADg37nUuMTgw?pwd=UANE) Code:UANE

We have released the codes of our UANet based on four backbones (*VGG, ResNet50, Res2Net-50, and PVT-v2-b2*). 

The whole training and testing framework of the paper have been released!

[![Video Thumbnail]([path/to/Jiepanli_Henry.github.io/images/video.png](https://www.bing.com/images/search?view=detailV2&ccid=cKLCfpL8&id=041FE5AA3ECBF94257350C5091C648EDBFDB3724&thid=OIP.cKLCfpL8GzUlBRgCzLBvlgHaEi&mediaurl=https%3a%2f%2fpic1.zhimg.com%2fv2-7f098d8b784c651ed3a24283c1068533_r.jpg&exph=930&expw=1517&q=%e5%bb%ba%e7%ad%91%e7%89%a9%e6%8f%90%e5%8f%96&simid=608029651695139278&FORM=IRPRST&ck=F57B6E5AD8D02AF7FBFEA73735C3B4FD&selectedIndex=0&itb=0&ajaxhist=0&ajaxserp=0))](https://henryjiepanli.github.io/Jiepanli_Henry.github.io/images/WeChat_20240124230813.mp4)
## Training Instructions

To train the UANet model, follow these steps:

1. Set the CUDA visible devices to specify the GPU for training. For example, to use GPU 0, run the following command:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python Code/train.py -c config/whubuilding/UANet.py
2. Set the CUDA visible devices to specify the GPU for test. For example, to use GPU 0, run the following command:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python Code/test.py -c config/whubuilding/UANet.py -o test_results/whubuilding/UANet/ --rgb

## Testing Instructions with Test Time Augmentation (TTA)

To perform testing with Test Time Augmentation (TTA), follow these steps:

1. Run the following command to perform testing with TTA:
   ```bash
   python Code/test.py -c config/whubuilding/UANet.py -o test_results/whubuilding/UANet/ -t lr --rgb

## Training Instructions for Multiple Training Sessions

If you want to continue training the model from a checkpoint or perform multiple training sessions, follow this:

1. Adjust the *pretrained_ckpt_path* in the config file.


## Reference
Our data processing  and whole framework are based on the [BuildFormer](https://github.com/WangLibo1995/BuildFormer). Here, we sincerely express our gratitude to the authors of that paper.

## Citation

We appreciate your attention to our work!

```bibtex
@ARTICLE{10418227,
  author={Li, Jiepan and He, Wei and Cao, Weinan and Zhang, Liangpei and Zhang, Hongyan},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={UANet: An Uncertainty-Aware Network for Building Extraction From Remote Sensing Images}, 
  year={2024},
  volume={62},
  number={},
  pages={1-13},
  keywords={Feature extraction;Uncertainty;Buildings;Data mining;Decoding;Remote sensing;Deep learning;Building extraction;remote sensing (RS);uncertainty-aware},
  doi={10.1109/TGRS.2024.3361211}}

