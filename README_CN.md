
# HCF-Net
## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>
HCFnet 是一个用于红外小目标分割的框架
## 数据集结构
如果你想要在自己的数据集上训练，你需要按照下列的结构准备数据:
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

使用下面的命令进行训练:

```train
python train.py --opt ./options/train.yaml
```
## Evaluation


使用下面的命令进行预测:

```eval
python test.py --opt ./options/test.yaml
```
## 预训练权重和结果
你可以下载预训练权重（我们还提供了整个训练的logs）:

- [HCF for SIRST](https://drive.google.com/drive/folders/1KljHLQjJVdMmaZXnkf1dtajtD8D28n7T?usp=drive_link)  

| Model name | IoU   | nIoU  |
|------------|-------|-------|
| UCF Net    | 80.09 | 78.31 |
