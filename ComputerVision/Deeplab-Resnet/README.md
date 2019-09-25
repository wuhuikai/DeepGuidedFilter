# Semantic Segmentation with Deeplab-Resnet-101

Mean IOU: **73.58%** [Baseline: 71.79%]

## Try it on an image!
1. Download the pretrained model [[Google Drive](https://drive.google.com/open?id=1YXZoZIZNR1ACewiUBp4UDvo_P65cCooK)|[BaiduYunPan](https://pan.baidu.com/s/1dEnpcGfchlZA_fVGdve0ig)].
2. Run it now!
    ```sh
    python predict_dgf.py --img_path [IM_PATH] --snapshots [MODEL_PATH]
    ```
## Prepare Dataset: PASCAL-VOC 2012
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
## Training
1. Download MS-COCO pretrained model [[Google Drive](https://drive.google.com/file/d/12ZLRUFQzmC7FFPZpS5tkzOQZLrhG7qt1/view?usp=sharing)|[BaiduYunPan](https://pan.baidu.com/s/1k0ODhkI65_h1szUamtGs0w)], put it in the folder [data](data).
2. Train the model!
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
## Acknowledgement
A part of the code was borrowed from [pytorch-deeplab-resnet](https://github.com/isht7/pytorch-deeplab-resnet).