import torch
import torch.nn as nn
import torch.utils.data as Data
import cv2
import numpy as np
from tqdm import tqdm
from basicseg.test_model import Test_model
from basicseg.utils.yaml_options import parse_options, dict2str
from basicseg.utils.path_utils import *
from basicseg.data import build_dataset

def init_dataset(opt):
    test_opt = opt['dataset']['test']
    testset = build_dataset(test_opt)
    return testset

def init_dataloader(opt, testset):
    test_loader  = Data.DataLoader(dataset=testset, batch_size=opt['exp']['bs'],
                                    sampler=None, num_workers=opt['exp'].get('nw', 8))
    return test_loader

def tensor2img(inp):
    # [b,1,h,w] -> [b,h,w]-> cpu -> numpy.array -> np.uint8
    # we don't do binarize here,
    # if you want to only contain 0 and 255, you can modify code here
    inp = torch.sigmoid(inp) * 255.
    inp = inp.squeeze(1).cpu().numpy().astype(np.uint8)
    return inp

def save_batch_img(imgs, img_names, dire):
    for i in range(len(imgs)):
        img = imgs[i]
        img_name = img_names[i]
        img_path = os.path.join(dire, img_name)
        cv2.imwrite(img_path, img)

def main():
    opt, args = parse_options()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt['exp']['device'])
    # init dataset
    testset = init_dataset(opt)
    test_loader = init_dataloader(opt, testset)
    # 初始化 模型参数, 包含 网络 优化器 损失函数 学习率准则
    # initialize parameters including network, optimizer, loss function, learning rate scheduler
    model = Test_model(opt)
    save_dir = opt['exp'].get('save_dir', False)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # load model params
    if opt.get('resume'):
        if opt['resume'].get('net_path'):
            model.load_network(model.net, opt['resume']['net_path'])
            print(f'load pretrained network from: {opt["resume"]["net_path"]}')

    model.net.eval()
    for idx, data in enumerate(tqdm(test_loader)):
        img, label, img_name = data
        with torch.no_grad():
            pred = model.test_one_iter((img, label))
        if save_dir:
            img_np = tensor2img(pred)
            save_batch_img(img_np, img_name, save_dir)
    test_mean_metric = model.get_mean_metric()
    test_norm_metric = model.get_norm_metric()
    ########## trainging done ##########
    print(f"best_mean_metric: [miou: {test_mean_metric['iou']:.4f}] [mfscore: {test_mean_metric['fscore']:.4f}]")
    print(f"best_norm_metric: [niou: {test_norm_metric['iou']:.4f}] [nfscore: {test_norm_metric['fscore']:.4f}]")

if __name__ == '__main__':
    main()