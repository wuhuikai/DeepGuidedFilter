# Fast End-to-End Trainable Guided Filter
[[Project]](http://wuhuikai.me/DeepGuidedFilterProject)    [[Paper]](http://wuhuikai.me/DeepGuidedFilterProject/deep_guided_filter.pdf)    [[arXiv]](https://arxiv.org/abs/1803.05619)    [[Demo]](http://wuhuikai.me/DeepGuidedFilterProject#demo)    [[Home]](http://wuhuikai.me)

Official implementation of **Fast End-to-End Trainable Guided Filter**.     
**Faster**, **Better** and **Lighter**  for image processing and dense prediction.

## Paper
**Fast End-to-End Trainable Guided Filter**     
Huikai Wu, Shuai Zheng, Junge Zhang, Kaiqi Huang    
CVPR 2018

## Install
```
pip install guided-filter-pytorch
```
## Usage
```
from guided_filter_pytorch.guided_filter import FastGuidedFilter

hr_y = FastGuidedFilter(r, eps)(lr_x, lr_y, hr_x)
```
```
from guided_filter_pytorch.guided_filter import GuidedFilter

hr_y = GuidedFilter(r, eps)(hr_x, init_hr_y)
``` 
## Citation
```
@inproceedings{wu2017fast,
  title     = {Fast End-to-End Trainable Guided Filter},
  author    = {Wu, Huikai and Zheng, Shuai and Zhang, Junge and Huang, Kaiqi},
  booktitle = {CVPR},
  year = {2018}
}
```