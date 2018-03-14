# DGF for Image Processing
## Prepare Dataset
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

## Training
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

## Evaluate
```sh
python test_hr.py --task [l0_smooth|multi_scale_detail_manipulation|style_transfer|non_local_dehazing|auto_ps] \
                  --name [HR|HR_AD|HR_FT|HR_AD_FT|HR_PP] \
                  --model [guided_filter|deep_guided_filter|deep_guided_filter_advanced]
```
**NOTE**:
* guided_filter: **DGF<sub>s</sub>**
* deep_guided_filter: **DGF<sub>b</sub>**
* deep_guided_filter_advanced: **DGF**

## Running Time
```sh
python test_time.py --model_id [0|1|2]
```
**NOTE**:
* 0: **DGF<sub>b</sub>**
* 1: GuidedFilteringLayer
* 2: **DGF**

## Predict
```sh
python predict.py  --task [l0_smooth|multi_scale_detail_manipulation|style_transfer|non_local_dehazing|auto_ps] \
                  [--img_path IMG_PATH | --img_list IMG_LIST] \
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
* --gray: It's better to generate gray images for style transfer.

## Acknowledgement
A part of the code has been adapted from [FastImageProcessing](https://github.com/CQFIO/FastImageProcessing).