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
pip install guided-filter-tf
```
## Usage
```
from guided_filter_tf.guided_filter import fast_guided_filter
    
hr_y = fast_guided_filter(lr_x, lr_y, hr_x, r, eps, nhwc)
```
```
from guided_filter_tf.guided_filter import guided_filter
hr_y = guided_filter(hr_x, init_hr_y, r, eps, nhwc)
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