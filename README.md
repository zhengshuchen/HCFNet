
# HCF-Net
## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>
HCF-Net is a framework for Infrared Small Object Segmentation
## Dataset Structe
If you want to train on custom datasets you should paper dataset as following structure:
```
|-SIRST
    |-trainval
        |-images
            |-xxx.png
        |-masks
            |-xxx.png
    |-test
        |-images
            |-xxx.png
        |-masks
            |-xxx.png
```
## Training

To train the model, run this command:

```train
python train.py --opt ./options/train.yaml
```
## Evaluation


To evaluate pretrained model, run:

```eval
python test.py --opt ./options/test.yaml
```
## Pre-trained Models and Results

You can download pretrained models (we also provide the whole training logs) here:

- [HCF for SIRST](https://drive.google.com/drive/folders/1KljHLQjJVdMmaZXnkf1dtajtD8D28n7T?usp=drive_link)


| Model name | IoU   | nIoU  |
|------------|-------|-------|
| UCF Net    | 80.09 | 78.31 |
