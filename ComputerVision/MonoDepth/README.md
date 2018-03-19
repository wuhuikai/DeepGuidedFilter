# Monocular Depth Estimation

RMS: **5.887** [Baseline: 6.081]

## Try it on an image !
1. **Download** and **Unzip** Pretrained Model
    * **WITH** Guided Filtering Layer [[Google Drive](https://drive.google.com/file/d/1dKDYRtZPahoFJZ5ZJNilgHEvT6gG4SC6/view?usp=sharing)|[BaiduYunPan](https://pan.baidu.com/s/1-GkMaRAVym8UEmQ6ia5cHw)]
    * **WITHOUT** Guided Filtering Layer [[Google Drive](https://drive.google.com/file/d/1w-f75x8WYRKukoQOP-TYJIq4--W40nrq/view?usp=sharing)|[BaiduYunPan](https://pan.baidu.com/s/19IkFGX5I-zc3Ap5mZHQzOw)]
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
## Training on KITTI
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
4. Download [Test Dataset](http://www.cvlibs.net/download.php?file=data_scene_flow.zip)
5. Testing
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
## Acknowledgement
A part of the code has been borrowed from [monodepth](https://github.com/mrharicot/monodepth).
