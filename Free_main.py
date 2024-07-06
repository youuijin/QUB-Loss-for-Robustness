# This module is adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import sys
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, 'lib')
import argparse, csv
import os
import time
import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import math
import numpy as np
from utils import *
# from validation import validate, validate_pgd
import torchvision.models as models

from utils import train_utils, data_utils
from datetime import datetime

import torch.nn.functional as F
from attack.IterativeAttack import PGDAttack

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

def main(args):
    ## set seed ##
    train_utils.set_seed()
    device = torch.device(f'cuda:{args.device_num}')
    
    cur = datetime.now().strftime('%m-%d_%H-%M')
    log_dir = f'results/{args.dataset}/{args.model}'
    log_setting = f'{args.loss}/Free(m{args.n_repeat})/eps{args.eps}(alpha{args.a2})/lr{args.lr}_multistep/{cur}'

    writer = SummaryWriter(f"{log_dir}/logs/{log_setting}")
    csv_path = f"{log_dir}/csvs/result.csv"

    # Create the model
    model = train_utils.set_model(args.model, args.n_way, args.imgc)
    model = model.to(device)
    
    # Optimizer:
    # optim, scheduler = train_utils.set_optim(model, args.sche, args.lr, args.epoch)
    tot_epochs = int(math.ceil(args.epoch/args.n_repeat))
    optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0002)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[int(tot_epochs*0.5), int(tot_epochs*0.8)], gamma=0.1) 
    
    ## use cifar10 dataloader
    train_loader, val_loader, test_loader = data_utils.set_dataloader(args)

    global global_noise_data
    global_noise_data = torch.zeros([args.batch_size, 3, args.imgsz, args.imgsz]).to(device)

    eps = args.eps/255.
    step_size = args.a2/255.
    pgd_attack = PGDAttack(model, eps=8.0, alpha=2.0, iter=10, restart=1, device=device)

    best_acc, train_time, attack_time = 0, 0, 0
    for epoch in tqdm(range(tot_epochs), desc='epoch', position=0):
        train_loss = 0

        # global global_noise_data
        model.train()
        loss_per_repeat = [0 for _ in range(args.n_repeat)]
        
        train_time_st = time.time()
        for i, (x, y) in enumerate(tqdm(train_loader, desc='train', position=1, ascii=" =", leave=False)):
            x, y = x.to(device), y.to(device)
            for j in range(args.n_repeat):
                optim.zero_grad()

                # Ascend on the global noise
                attack_time_st = time.time()
                noise_batch = Variable(global_noise_data[0:x.size(0)], requires_grad=True).to(device)
                x1 = (x + noise_batch).detach()
                x1 = torch.clamp(x1, 0, 1)
                x1.requires_grad_()

                attack_time += (time.time() - attack_time_st)
                if args.loss == 'CE':
                    adv_logit = model(x1)
                    loss = F.cross_entropy(adv_logit, y) # Cross entropy loss
                elif args.loss == 'QUB':
                    logit = model(x)
                    softmax = F.softmax(logit, dim=1)
                    y_onehot = F.one_hot(y, num_classes = softmax.shape[1])
                    adv_logit = model(x1)
                    adv_norm = torch.norm(adv_logit-logit, dim=1)
                    loss = F.cross_entropy(logit, y, reduction='none')
                    loss = loss + torch.sum((adv_logit-logit)*(softmax-y_onehot), dim=1) + 0.5/2.0*torch.pow(adv_norm, 2)
                    loss = loss.mean()

                train_loss += loss.item()*x.shape[0]
                loss_per_repeat[j] += loss.item()*x.shape[0]

                # compute gradient and do SGD step
                
                loss.backward()
                optim.step()
                
                # Update the noise for the next iteration
                '''Original'''
                # pert = noise_batch.grad * step_size
                # global_noise_data[0:x.size(0)] += pert.data
                # global_noise_data.clamp_(-eps, eps)
                '''Another github'''
                grad = x1.grad.data
                global_noise_data[0:x.size(0)] = noise_batch.detach() + step_size * torch.sign(grad.detach())
                global_noise_data.clamp_(-eps, eps)

                writer.add_scalar("train/step_loss", round(loss.item(), 4), epoch*args.n_repeat*len(train_loader) + i*args.n_repeat + j)

        train_time += (time.time() - train_time_st)

        writer.add_scalar("train/acc", 0, epoch)
        writer.add_scalar("train/loss", round(train_loss/len(train_loader.dataset)/args.n_repeat, 4), epoch)
        for r in range(args.n_repeat):
            writer.add_scalar("train/repeat_loss", round(loss_per_repeat[r]/len(train_loader.dataset), 4), epoch*args.n_repeat + r)

        if epoch%2==0:
            model.eval()
            val_correct_count, val_adv_correct_count = 0, 0
            val_loss, val_adv_loss = 0, 0
            for _, (x, y) in enumerate(tqdm(val_loader, desc='val step', position=1, leave=False, ascii=" *")):
                x, y = x.to(device), y.to(device)
                logit = model(x)
                pred = F.softmax(logit, dim=1)
                outputs = torch.argmax(pred, dim=1)
                val_loss += F.cross_entropy(logit, y).item()*x.shape[0]
                val_correct_count += (outputs == y).sum().item()

                advx = pgd_attack.perturb(x, y)
                adv_logit = model(advx)
                adv_pred = F.softmax(adv_logit, dim=1)
                adv_outputs = torch.argmax(adv_pred, dim=1)
                val_adv_loss += F.cross_entropy(adv_logit, y).item()*x.shape[0]
                val_adv_correct_count += (adv_outputs == y).sum().item()

            writer.add_scalar("val/acc", round(val_correct_count/len(val_loader.dataset)*100, 4), epoch)
            writer.add_scalar("val/loss", round(val_loss/len(val_loader.dataset), 4), epoch)
            writer.add_scalar("val/acc_adv", round(val_adv_correct_count/len(val_loader.dataset)*100, 4), epoch)
            writer.add_scalar("val/loss_adv", round(val_adv_loss/len(val_loader.dataset), 4), epoch)

            last_val = round(val_correct_count/len(val_loader.dataset)*100, 4)
            last_val_adv = round(val_adv_correct_count/len(val_loader.dataset)*100, 4)

            if best_acc < round(val_adv_correct_count/len(val_loader.dataset)*100, 4):
                best_acc = round(val_adv_correct_count/len(val_loader.dataset)*100, 4)
                best_accs = [round(val_correct_count/len(val_loader.dataset)*100, 4), round(val_adv_correct_count/len(val_loader.dataset)*100, 4)]
                torch.save(model.state_dict(), f'{log_dir}/saved_model/{args.model}_{log_setting.replace("/", "_")}.pt')

        scheduler.step()
    torch.save(model.state_dict(), f'{log_dir}/saved_model/{args.model}_{log_setting.replace("/", "_")}_last.pt')

    # # test
    # model.eval()
    # test_correct_count = 0
    # test_adv_correct_count = 0
    # # val_attack = attack_utils.set_attack(self.args.val_attack, self.model, self.args.val_eps, self.args)
    # with torch.no_grad():
    #     for _, (x, y) in enumerate(tqdm(test_loader, desc="test")):
    #         x, y = x.to(device), y.to(device)
    #         logit = model(x)
    #         pred = F.softmax(logit, dim=1)
    #         outputs = torch.argmax(pred, dim=1)
    #         test_correct_count += (outputs == y).sum().item()
    #         advx = pgd_attack.perturb(x, y)
    #         adv_logit = model(advx)
    #         adv_pred = F.softmax(adv_logit, dim=1)
    #         adv_outputs = torch.argmax(adv_pred, dim=1)
    #         test_adv_correct_count += (adv_outputs == y).sum().item()

    # test_acc = round(test_correct_count/len(test_loader.dataset)*100, 4)
    # test_adv_acc = round(test_adv_correct_count/len(test_loader.dataset)*100, 4)

    # writer.add_scalar("test/acc", test_adv_acc*100, test_acc*100)

    result = [log_setting, best_accs[0], best_accs[1], last_val, last_val_adv, round(train_time, 4), round(attack_time, 4)]
    with open(csv_path, 'a', encoding='utf-8', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(result)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, help='dataset', default='cifar10')
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=32)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--n_way', type=int, help='n way', default=10)
    argparser.add_argument('--normalize', type=str, help='normalize range', default='none')

    ## Model options
    argparser.add_argument('--model', type=str, help='type of model to use', default="resnet18")
    argparser.add_argument('--device_num', type=int, help='which gpu to use', default=0)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=150)
    argparser.add_argument('--batch_size', type=int, help='batch size in epoch', default=128)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.1)
    argparser.add_argument('--sche', type=str, default="multistep")
    argparser.add_argument('--loss', type=str, default='CE')

    argparser.add_argument('--eps', type=float, default=8.)
    argparser.add_argument('--a2', type=float, default=8.)

    argparser.add_argument('--n_repeat', type=int, default=8)

    args = argparser.parse_args()

    main(args)

'''
TRAIN:
    # Number of training epochs
    epochs: 90
    
    # Architecture name, see pytorch models package for
    # a list of possible architectures
    arch: 'resnet50'

    # Starting epoch
    start_epoch: 0

    # SGD paramters
    lr: 0.1
    momentum: 0.9
    weight_decay: 0.0001

    # Print frequency, is used for both training and testing
    print_freq: 10

    # Dataset mean and std used for data normalization
    mean: !!python/tuple [0.485, 0.456, 0.406]
    std: !!python/tuple [0.229, 0.224, 0.225]
    
ADV:
    # FGSM parameters during training
    clip_eps: 4.0
    fgsm_step: 4.0

    # Number of repeats for free adversarial training
    n_repeat: 4

    # PGD attack parameters used during validation
    # the same clip_eps as above is used for PGD
    pgd_attack: 
    - !!python/tuple [10, 0.00392156862] #[10 iters, 1.0/255.0]
    - !!python/tuple [50, 0.00392156862] #[50 iters, 1.0/255.0]
    
DATA:
    # Number of data workers
    workers: 4

    # Training batch size
    batch_size: 256

    # Image Size
    img_size: 256

    # Crop Size for data augmentation
    crop_size: 224

    # Color value range
    max_color_value: 255.0

'''