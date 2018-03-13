# Fast End-to-End Trainable Guided Filter
[[Project]]()    [[Paper]]()    [[Demo]]()    [[Home]]()  
Official implementation of **Fast End-to-End Trainable Guided Filter**.

**Faster**, **Better** and **Lighter**  for image processing and dense prediction. 

## Overview
![](images/results.jpg)

**DeepGuidedFilter** is the author's implementation of the deep learning building block for joint upsampling described in:  
"Fast End-to-End Trainable Guided Filter"   
Huikai Wu, Shuai Zheng, Junge Zhang, Kaiqi Huang

Given a high-resolution image, a corresponding low-resolution image and a low-resolution target, our algorithm generates the corresponding high-resolution target. Through joint training with CNNs, our algorithm achieves the state-of-the-art performance while runs **10-100** times faster. 

Contact: Hui-Kai Wu (huikaiwu@icloud.com)

## Try it on an image !
### Install dependencies

### Install Guided Filtering Layer
### Install
#### Released Version
* PyTorch Version
    ```sh
    pip install guided-filter-pytorch
    ```
* Tensorflow Version
    ```sh
    pip install guided-filter-tf
    ```
#### Latest Version
* PyTorch Version
    ```sh
    pip install -e GuidedFilteringLayer/GuidedFilter_PyTorch
    ```
* Tensorflow Version
    ```sh
    pip install -e GuidedFilteringLayer/GuidedFilter_TF
    ```