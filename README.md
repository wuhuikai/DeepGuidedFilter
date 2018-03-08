## DGF for Image Processing
### Training


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
    python split.py --set 512    --task [l0_smooth|multi_scale_detail_manipulation|style_transfer|non_local_dehazing|auto_ps]
    python split.py --set 1024   --task [l0_smooth|multi_scale_detail_manipulation|style_transfer|non_local_dehazing|auto_ps]
    python split.py --set random --task [l0_smooth|multi_scale_detail_manipulation|style_transfer|non_local_dehazing|auto_ps]
    ```