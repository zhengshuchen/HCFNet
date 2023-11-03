
# HCFnet
## <div align="center"><b><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></b></div>
HCFnet is a framework for Infrared Small Object Segmentation, 
Which supports:
- [x] Training
- [x] Resume training from checkpoint 
- [x] Multi-GPUs Training (DDP)
- [x] Test and inference

It also including the official implementation of our paper
[Local Contrast and Global Contextual Information Make 
Infrared Small Object 
Salient Again](https://arxiv.org/abs/2301.12093)




## Requirements
numpy >= 1.21.2 (maybe lower is ok)

opencv >= 4.5.3 (maybe lower is ok)

pytorch >= 1.9.1 (maybe >= 1.8.1 is also ok)

albumentations >= 0.5.2

etc. 


>📋  Note: The whole environment we use is in requirements.txt,
> but you don't have to use exactly the same version as we use.

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
python train.py --opt ./options/hcf_train.yaml
```

>📋 Once you run this train command it will automatically create a folder under ./experiments
> and save all the logs, tensorboard for losses and metrics, and the
> model params and training states.
## Evaluation


To evaluate pretrained model, run:

```eval
python test.py --opt ./test_options/hcf_test.yaml
```

>📋  usage example:
> 1. download and unzip datasets 
> 2. download pretrained model
> 3. modified the config file ./test_options/test_demo.yaml, specifically: 
> device (cpu are not support right now), data_root and net_path in your own case 

## Pre-trained Models and Results

You can download pretrained models (we also provide the whole training logs) here:

- [UCF for SIRST](https://drive.google.com/file/d/1JHdASkGF8Gefmw3C3Ar1fiIkM24bezIl/view?usp=share_link)


| Model name | IoU   | nIoU  |
|------------|-------|-------|
| UCF Net    | 80.89 | 78.72 |
