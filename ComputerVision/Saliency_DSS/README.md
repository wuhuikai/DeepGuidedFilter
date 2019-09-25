# Saliency Detection with DSS

Max F-measure: **91.75%** [Baseline: 90.61%]

## Try on an image!
1. Download the pretrained model [[Google Drive](https://drive.google.com/open?id=1ZxbAAJw9BxCKj2e2QsBmCnjWLFlCGLf1)|[BaiduYunPan](https://pan.baidu.com/s/1pgOMh3V50lRa6slbIW_SKQ)].
2. Try it now!
    ```sh
    python predict.py --im_path [IM_PATH] \
                      --netG [MODEL_PATH] \
                      --thres [-1|161] \
                      --dgf --nn_dgf \
                      --post_sigmoid --cuda
    ```
## Training on MSRA-B
1. Download and **unzip** the saliency dataset [MSRA-B](http://mftp.mmcheng.net/Data/MSRA-B.zip).
2. Make **A-B** (Image-Label) pairs.
    ```sh
    python scripts/preprocess.py --data_path [MSRA-B_ROOT] --mode train
    python scripts/preprocess.py --data_path [MSRA-B_ROOT] --mode valid
    python scripts/preprocess.py --data_path [MSRA-B_ROOT] --mode test
    ```
3. Start training!
    * **WITHOUT** Guided Filtering Layer
    ```sh
    python main.py --dataroot [MSRA-B_ROOT]/AB --cuda --experiment [EXP_NAME]
    ```
    * **WITH** Guided Filtering Layer
    ```sh
    python main.py --dataroot [MSRA-B_ROOT]/AB --cuda --experiment [EXP_NAME] --dgf
    ```
    * Finetune
    ```sh
    python main.py --dataroot [MSRA-B_ROOT]/AB --cuda --experiment [EXP_NAME] --netG [MODEL_PATH] --dgf
    ```
    
4. Evaluation
    ```sh
    python test.py --dataroot [MSRA-B_ROOT]/AB/test \
                   --netG [MODEL_PATH] --cuda \
                   --experiment [SAVE_FOLDER] \
                   --nn_dgf --post_sigmoid --dgf
    ```
5. Calculate metrics
    1. Install [SalMetric](https://github.com/Andrew-Qibin/SalMetric).
        ```sh
        git clone https://github.com/Andrew-Qibin/SalMetric && cd SalMetric 
        mkdir build && cd build
        cmake .. && make
        ```
    2. Calculate !
       ```sh
       cd [SAVE_FOLDER]
       [SalMetric_ROOT]/build/salmetric test.txt [WORKER_NUM]
       ```
## Acknowledgement
A part of the code was adapted from [DSS](https://github.com/Andrew-Qibin/DSS).