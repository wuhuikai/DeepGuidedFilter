# Fast End-to-End Trainable Guided Filter
[[Project]]()    [[Paper]]()    [[Demo]]()    [[Home]]()
  
Official implementation of **Fast End-to-End Trainable Guided Filter**.     
**Faster**, **Better** and **Lighter**  for image processing and dense prediction.

**Fast End-to-End Trainable Guided Filter**     
Huikai Wu, Shuai Zheng, Junge Zhang, Kaiqi Huang    
CVPR 2018

## Guided Filtering Layer
### Install Released Version
```sh
pip install guided-filter-pytorch
```
### Usage
```python
from guided_filter_pytorch.guided_filter import FastGuidedFilter

hr_y = FastGuidedFilter(r, eps)(lr_x, lr_y, hr_x)
```
```python
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