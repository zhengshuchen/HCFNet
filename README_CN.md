
# HCFnet
## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>
HCFnet 是一个用于红外弱小目标分割的框架, 
目前支持:
- [x] 单卡训练
- [x] 从断点续训练 
- [x] 多卡训练 (采用DDP模式)
- [x] 测试和推理

它同时包含了我们文章的官方实现
[Local Contrast and Global Contextual Information Make 
Infrared Small Object 
Salient Again](https://arxiv.org/abs/2301.12093)



## 环境需要
numpy >= 1.21.2 (maybe lower is ok)

opencv >= 4.5.3 (maybe lower is ok)

pytorch >= 1.9.1 (maybe >= 1.8.1 is also ok)

albumentations >= 0.5.2

etc. 


>📋  注：我们所使用的所有环境在requirements.txt里，但并不一定要完全与我们所使用的环境版本相同。

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
python train.py --opt ./options/hcf_train.yaml
```

>📋 使用代码进行训练后，它会自动在./experiments目录下创建一个文件件，里面会保存这次实验的所有logs，损失值和评价指标的tensorboard曲线，
> 模型参数和训练状态。
## Evaluation


使用下面的命令进行预测:

```eval
python test.py --opt ./test_options/ucf_test.yaml
```

>📋  简单例子:
> 1. 下载并解压数据集 
> 2. 下载预训练权重
> 3. 更改配置文件./test_options/test_demo.yaml, 具体地:你需要改使用的
> device (cpu are not support right now), data_root and net_path使其符合你自己的环境。 

## 预训练权重和结果
你可以下载预训练权重（我们还提供了整个训练的logs）:

- [UCF for SIRST](https://drive.google.com/mymodel.pth)  

| Model name | IoU   | nIoU  |
|------------|-------|-------|
| UCF Net    | 80.89 | 78.72 |
