# Fast End-to-End Trainable Guided Filter
[[Project]]()    [[Paper]]()    [[Demo]]()    [[Home]]()  
Official implementation of **Fast End-to-End Trainable Guided Filter**.

**Faster**, **Better** and **Lighter**  for image processing and dense prediction. 

## Overview
![](images/results.jpg)

**DeepGuidedFilter** is the author's implementation of the deep learning building block for joint upsampling described in:  
```
Fast End-to-End Trainable Guided Filter   
Huikai Wu, Shuai Zheng, Junge Zhang, Kaiqi Huang
```

Given a high-resolution image, a corresponding low-resolution image and a low-resolution target, our algorithm generates the corresponding high-resolution target. Through joint training with CNNs, our algorithm achieves the state-of-the-art performance while runs **10-100** times faster. 

Contact: Hui-Kai Wu (huikaiwu@icloud.com)

## Try it on an image !
### Prepare Environment
1. Download source code from GitHub.
    ```sh
    git clone https://github.com/wuhuikai/DeepGuidedFilter
    
    cd DeepGuidedFilter && git checkout release
    ```
2. Install dependencies (PyTorch version).
    ```sh
    conda install opencv
    conda install pytorch=0.2.0 cuda80 -c soumith
    
    pip install -r requirements.txt 
    ```
3. (**Optional**) Install dependencies for MonoDepth (Tensorflow version).
    ```sh
    cd ComputerVision/MonoDepth
    
    pip install -r requirements.txt
    ```
### Ready to **GO** !
