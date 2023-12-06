
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
## Trained Results

We provide the whole training logs：
```
train_HCF_demo_20231112_201631.log
```
| Model name | IoU   | nIoU  |
|------------|-------|-------|
| UCF Net    | 80.09 | 78.31 |
