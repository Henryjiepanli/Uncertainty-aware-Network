# Codes for 《 UANet: an Uncertainty-Aware Network for Building Extraction from Remote Sensing Images》

## About Paper
We are delighted to inform everyone that our paper has been successfully accepted by TGRS (IEEE Transactions on Geoscience and Remote Sensing). 
[Paper Link](https://ieeexplore.ieee.org/document/10418227)

The results on the three building datasets can be downloaded via Baidu Disk：[Link](https://pan.baidu.com/s/1MkoWfIyz7DADg37nUuMTgw?pwd=UANE) Code:UANE

We have released the codes of our UANet based on four backbones (*VGG, ResNet50, Res2Net-50, and PVT-v2-b2*). 

The whole training and testing framework of the paper have been released!

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



## Reference
Our data processing is based on the [BuildFormer](https://github.com/WangLibo1995/BuildFormer). Here, we sincerely express our gratitude to the authors of that paper.

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

