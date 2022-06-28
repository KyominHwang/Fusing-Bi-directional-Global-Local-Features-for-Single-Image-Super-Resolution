import argparse
import os
import sys

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import SRModel
from dataset import *
from utils import *
import torchvision.transforms as transforms

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_path', type=str, required=True)
    parser.add_argument('--scale_factor', type=int, required=True)
    parser.add_argument('--benchmark_gt_path', type=str, required=True)
    parser.add_argument('--benchmark_lr_path', type=str, required=True)
    parser.add_argument('--down_sample', type=int, default=2)
    parser.add_argument('--img_range', type=int, default=1.)
    parser.add_argument('--n_feats', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--n_blocks', type=int, default=10)
    parser.add_argument('--n_res_blocks', type=int, default=10)
    parser.add_argument('--n_workers', type=int, default=4)
    opt = parser.parse_args()

    model = SRModel(opt).to(device)
    
    valid_dataset = Dataset(opt.benchmark_lr_path, opt.benchmark_gt_path)
    valid_loader = DataLoader(valid_dataset, batch_size = 1, num_workers = 2, shuffle = True)
    
    m = torch.load(opt.pth_path)
    model.load_state_dict(m["model_state_dict"])
    
    best_psnr = 0
    model.eval()
    psnr_list = []
    with torch.no_grad():
        for _, val_data in enumerate(valid_loader):
            lr, gt = val_data["LR"].to(device), val_data["GT"].to(device)
            bs, c, h, w = lr.size()
            output = model(lr)
            output = output.squeeze().cpu().numpy()
            output = np.clip(output, 0.0, 1.0)
            gt = gt.squeeze().cpu().numpy()
            psnr = calculate_psnr(output * 255., gt * 255., input_order = 'CHW', test_y_channel = True, crop_border = 2)
            psnr_list.append(psnr)
    val_psnr = np.mean(psnr_list)
    print(f"valid psnr : {val_psnr}")