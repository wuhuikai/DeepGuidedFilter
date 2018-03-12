## DGF for Computer Vision


## Guided Filtering Layer
### Install
* PyTorch Version
    ```sh
    pip install -e GuidedFilteringLayer/GuidedFilter_PyTorch
    ```
* Tensorflow Version
    ```sh
    pip install -e GuidedFilteringLayer/GuidedFilter_TF
    ```

### Usage
* PyTorch Version
    ```python
    from guided_filter_pytorch.guided_filter import FastGuidedFilter
    
    hr_y = FastGuidedFilter(r, eps)(lr_x, lr_y, hr_x)
    ```
    ```python
    from guided_filter_pytorch.guided_filter import GuidedFilter
    
    hr_y = GuidedFilter(r, eps)(hr_x, init_hr_y)
    ``` 
* Tensorflow Version
    ```python
    from guided_filter_tf.guided_filter import fast_guided_filter
    
    hr_y = fast_guided_filter(lr_x, lr_y, hr_x, r, eps, nhwc)
    ```
    ```python
    from guided_filter_tf.guided_filter import guided_filter
    
    hr_y = guided_filter(hr_x, init_hr_y, r, eps, nhwc)
    ``` 
    
## DGF for Image Processing
### Prepare Dataset
1. Download MIT-Adobe FiveK [Dataset](https://data.csail.mit.edu/graphics/fivek/fivek_dataset.tar).
    ```sh
    dataset
    ├── ......
    └── fivek
        ├── ......
        └── raw_photos
            ├── HQa1to700
            │   └── photos
            │       ├── a0001-jmac_DSC1459.dng
            │       └── ......
            └── ......
    ```
2. Convert from ***.dng** to ***.tif**.
    ```sh
    cd scripts
    python convert_dng_to_tif.py
    ```
3. Generate Training/Test set with different resolution.
    ```sh
    cd scripts
    
    python precompute_size.py --min 512
    python precompute_size.py --min 1024
    python precompute_size.py --random
    
    python resize_image.py --file_name 512
    python resize_image.py --file_name 1024
    python resize_image.py --file_name random
    ```
4. Generate Ground Truth for each task.
    * L<sub>0</sub> smoothing / Multi-scale detail manipulation / Style transfer / Non-local dehazing
        ```sh
        cd scripts/[l0_smooth|multi_scale_detail_manipulation|style_transfer|non_local_dehazing]
        
        matlab -r "prepare_dataset(   '512'); exit" -nodisplay
        matlab -r "prepare_dataset(  '1024'); exit" -nodisplay
        matlab -r "prepare_dataset('random'); exit" -nodisplay
        ```
    * Image retouching (Auto-PS, Auto Beautification)
        1. Generate / Download Ground Truth
            ```sh
            dataset
            ├── ......
            └── fivek
                ├── ......
                └── gts
                    ├── a0001-jmac_DSC1459.tif
                    └── ......
            ```
            * **Option 1**: Generate ground truth with Adobe Lightroom by following the [instruction](https://data.csail.mit.edu/graphics/fivek/).
            * **Option 2 (Preferred)**: Download from the website.
                ```sh
                cd scripts/auto_ps
                
                # Expert [A|B|C|D|E]
                bash generate_list.sh [a|b|c|d|e]
                
                cd ../../dataset/fivek && mkdir gts && cd gts
                # Option 1: single thread
                wget -i ../../../scripts/auto_ps/img_list_[a|b|c|d|e].csv
                # Option 2: multi thread
                sudo apt-get install aria2
                aria2c -j 100 -i ../../../scripts/auto_ps/img_list_[a|b|c|d|e].csv
                ```
        2. Postprocess
            ```sh
            cd scripts/auto_ps
            python postprocess.py
            
            cd ..
            python resize_image.py --file_name 512    --task auto_ps
            python resize_image.py --file_name 1024   --task auto_ps
            python resize_image.py --file_name random --task auto_ps
            ```
5. Split Training/Test set
    ```sh
    cd scripts/training_test_split
    python split.py --set 512 \
                    --task [l0_smooth|multi_scale_detail_manipulation|style_transfer|non_local_dehazing|auto_ps]
    python split.py --set 1024 \
                    --task [l0_smooth|multi_scale_detail_manipulation|style_transfer|non_local_dehazing|auto_ps]
    python split.py --set random \
                    --task [l0_smooth|multi_scale_detail_manipulation|style_transfer|non_local_dehazing|auto_ps]
    ```

### Training
* Option 1: Train from scratch
    ```sh
    python train_hr.py --task [l0_smooth|multi_scale_detail_manipulation|style_transfer|non_local_dehazing|auto_ps] \
                       --name [HR|HR_AD] \
                       --model [deep_guided_filter|deep_guided_filter_advanced]
    ```
* Option 2: Train with low-resolution data + Finetune
    ```sh
    python train_lr.py --task [l0_smooth|multi_scale_detail_manipulation|style_transfer|non_local_dehazing|auto_ps]
    
    python train_hr_finetune.py --task [l0_smooth|multi_scale_detail_manipulation|style_transfer|non_local_dehazing|auto_ps] \
                                --name [HR_FT|HR_AD_FT] \
                                --model [deep_guided_filter|deep_guided_filter_advanced]
    ```
**NOTE**:
* deep_guided_filter: **DGF<sub>b</sub>**
* deep_guided_filter_advanced: **DGF**

### Evaluate
```sh
python test_hr.py --task [l0_smooth|multi_scale_detail_manipulation|style_transfer|non_local_dehazing|auto_ps] \
                  --name [HR|HR_AD|HR_FT|HR_AD_FT|HR_PP] \
                  --model [guided_filter|deep_guided_filter|deep_guided_filter_advanced]
```
**NOTE**:
* guided_filter: **DGF<sub>s</sub>**
* deep_guided_filter: **DGF<sub>b</sub>**
* deep_guided_filter_advanced: **DGF**

### Running Time
```sh
python test_time.py --model_id [0|1|2]
```
**NOTE**:
* 0: **DGF<sub>b</sub>**
* 1: GuidedFilteringLayer
* 2: **DGF**

### Predict
```sh
python predict.py [--img_path IMG_PATH | --img_list IMG_LIST] \
                   --model_path MODEL_PATH \
                   --save_folder SAVE_FOLDER \
                   --model [guided_filter|deep_guided_filter|deep_guided_filter_advanced] \
                   --low_size 64 \
                   --gpu 0 \
                  [--gray]
```
**NOTE**:
* --model
    * guided_filter: **DGF<sub>s</sub>**
    * deep_guided_filter: **DGF<sub>b</sub>**
    * deep_guided_filter_advanced: **DGF**
* --model_path: **ALL MODEL**s in the folder [models](models).
* --gray: It's better to generate gray images for style transfer.
## DGF for Computer Vision
### Monocular Depth Estimation
#### Try it on an image !
1. **Download** and **Unzip** Pretrained Model
    
    [[**WITH**](https://drive.google.com/file/d/1dKDYRtZPahoFJZ5ZJNilgHEvT6gG4SC6/view?usp=sharing)|[**WITHOUT**](https://drive.google.com/file/d/1w-f75x8WYRKukoQOP-TYJIq4--W40nrq/view?usp=sharing)] Guided Filtering Layer
2. Run on an Image !
    * **WITHOUT** Guided Filtering Layer
    ```sh
    python monodepth_simple.py --image_path [IMAGE_PATH] --checkpoint_path [MODEL_PATH]
    ```
    * **WITH** Guided Filter as **PostProcessing**
    ```sh
    python monodepth_simple.py --image_path [IMAGE_PATH] --checkpoint_path [MODEL_PATH] --guided_filter_eval
    ```
    * **WITH** Guided Filtering Layer
    ```sh
    python monodepth_simple.py --image_path [IMAGE_PATH] --checkpoint_path [MODEL_PATH] --guided_filter
    ```
#### Training on KITTI
1. Download KITTI
    ```sh
    wget -i utils/kitti_archives_to_download.txt -P [SAVE_FOLDER]
    ```
2. (**Optional**) Convert *.**png** to *.**jpg** to save space.
    ```sh
    find [SAVE_FOLDER] -name '*.png' | parallel 'convert {.}.png {.}.jpg && rm {}'
    ```
3. Let's train the model !
    * **WITHOUT** Guided Filtering Layer
        ```sh
        python monodepth_main.py --mode train \
                                 --model_name monodepth_kitti \
                                 --data_path [SAVE_FOLDER] \
                                 --filenames_file utils/filenames/kitti_train_files.txt \
                                 --log_directory checkpoints
        ```
    * **WITH** Guided Filtering Layer
        ```sh
        python monodepth_main.py --mode train \
                                 --model_name monodepth_kitti_dgf \
                                 --data_path [SAVE_FOLDER] \
                                 --filenames_file utils/filenames/kitti_train_files.txt \
                                 --log_directory checkpoints \
                                 --guided_filter
        ```
4. Testing
    * **WITHOUT** Guided Filtering Layer
        ```sh
        python monodepth_main.py --mode test \
                                 --data_path [SAVE_FOLDER] \
                                 --filenames_file utils/filenames/kitti_stereo_2015_test_files.txt \
                                 --checkpoint_path [MODEL_PATH]
        ``` 
    * **WITH** Guided Filter as **PostProcessing**
        ```sh
        python monodepth_main.py --mode test \
                                 --data_path [SAVE_FOLDER] \
                                 --filenames_file utils/filenames/kitti_stereo_2015_test_files.txt \
                                 --checkpoint_path [MODEL_PATH] \
                                 --guided_filter_eval
        ```
    * **WITH** Guided Filtering Layer
        ```sh
        python monodepth_main.py --mode test \
                                 --data_path [SAVE_FOLDER] \
                                 --filenames_file utils/filenames/kitti_stereo_2015_test_files.txt \
                                 --checkpoint_path [MODEL_PATH] \
                                 --guided_filter
        ```
5. Evaluation on KITTI
    ```sh
    python utils/evaluate_kitti.py --split kitti --predicted_disp_path [disparities.npy] --gt_path [SAVE_FOLDER]
    ```
### Semantic Segmentation with Deeplab-Resnet
#### Try it on an image!
1. Download the pretrained [model](https://drive.google.com/open?id=1YXZoZIZNR1ACewiUBp4UDvo_P65cCooK).
2. Run it now !
    ```sh
    python predict_dgf.py --img_path [IM_PATH] --snapshots [MODEL_PATH]
    ```
#### Prepare Dataset: PASCAL-VOC 2012
1. Download and unzip [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and [SBD](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz).
    ```sh
    ROOT
    ├── benchmark_RELEASE
    └── VOCdevkit
    ```
2. Convert *.**mat** to *.**png** for SBD.
    ```sh
    python scripts/convert_mat_to_png.py --sbd_path [ROOT]/benchmark_RELEASE
    ```
3. Convert labels for PASCAL VOC 2012.
    ```sh
    python scripts/convert_labels.py \
                [ROOT]/VOCdevkit/VOC2012/SegmentationClass \
                [ROOT]/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt \
                [ROOT]/VOCdevkit/VOC2012/SegmentationClass_1D
    ```
4. Combine **PASCAL VOC 2012** and **SBD**.
    ```sh
    cd [ROOT]
    mv VOCdevkit/VOC2012/SegmentationClass_1D/*.png benchmark_RELEASE/dataset/cls_png/
    mv VOCdevkit/VOC2012/JPEGImages/*.jpg benchmark_RELEASE/dataset/img/
    ```
5. Soft link.
    ```sh
     ln -s [ROOT]/benchmark_RELEASE/dataset/cls_png data/gt
     ln -s [ROOT]/benchmark_RELEASE/dataset/img data/img
    ```
#### Training
1. Download pretrained [model](https://drive.google.com/file/d/12ZLRUFQzmC7FFPZpS5tkzOQZLrhG7qt1/view?usp=sharing) on MS-COCO, put it in the folder [data](data).
2. Train the model !
    * **WITHOUT** Guided Filtering Layer
        ```sh
        python train_dgf.py --snapshots snapshots
        ```
    * **WITH** Guided Filtering Layer
        ```sh
        python train_dgf.py --dgf --snapshots snapshots_dgf
        ```
    * Finetune
        ```sh
        python train_dgf.py --dgf --snapshots snapshots_dgf_ft --ft --ft_model_path [MODEL_PATH]
        ```
    
3. Evaluation
    * **WITHOUT** Guided Filtering Layer
        ```sh
        python evalpyt_dgf.py --exp [SAVE_FOLDER] --snapshots [MODEL_PATH]
        ```
    * **WITH** Guided Filtering Layer
        ```sh
        python evalpyt_dgf.py --exp [SAVE_FOLDER] --snapshots [MODEL_PATH] --dgf
        ```