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
    parser.add_argument('--arch', type=str, default='BI')
    parser.add_argument('--LR_path', type=str, required=True)
    parser.add_argument('--GT_path', type=str, required=True)
    parser.add_argument('--outputs_dir', type=str, required=True)
    parser.add_argument('--scale_factor', type=int, required=True)
    parser.add_argument('--benchmark_gt_path', type=str, required=True)
    parser.add_argument('--benchmark_lr_path', type=str, required=True)
    parser.add_argument('--down_sample', type=int, default=2)
    parser.add_argument('--img_range', type=int, default=1.)
    parser.add_argument('--n_feats', type=int, default=64)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--n_blocks', type=int, default=10)
    parser.add_argument('--n_res_blocks', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=12000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_workers', type=int, default=4)
    #parser.add_argument('--seed', type=int, default=123)
    #parser.add_argument('--use_fast_loader', action='store_true')
    opt = parser.parse_args()

    prev_epoch = 0

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    #torch.manual_seed(opt.seed)

    model = SRModel(opt).to(device)
    criterion = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.5, milestones=[1250,2500,2750,500,5500])
    
    transform = transforms.Compose([crop(opt.scale_factor, opt.patch_size), augmentation()])

    train_dataset = Dataset(opt.LR_path, opt.GT_path, transform)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.n_workers)
    
    valid_dataset = Dataset(opt.benchmark_lr_path, opt.benchmark_gt_path)
    valid_loader = DataLoader(valid_dataset, batch_size = 1, num_workers = 2, shuffle = True)
    
    best_psnr = 0
    
    for e in range(prev_epoch, opt.num_epochs):
        model.train()
        train_loss = 0
        count = 0
        pbar = tqdm(enumerate(train_loader), file=sys.stdout)
        for _, tr_data in pbar:
            data, target = tr_data['LR'].to(device), tr_data['GT'].to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            count += 1
            pbar.set_postfix(epoch=f'epoch : {e + 1} of {opt.num_epochs}, loss : {train_loss / count}')
        
        scheduler.step()
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
        print(f"valid psnr : {val_psnr} epoch : {e}")
        if best_psnr < val_psnr:
            torch.save(model.state_dict(), os.path.join(opt.outputs_dir, '{}_epoch_{}.pth'.format(opt.arch, e)))
