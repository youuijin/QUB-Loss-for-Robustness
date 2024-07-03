import random
import  torch
import  numpy as np

from utils.model.resnet import *
from utils.model.wrn import *
from utils.SGDR import CosineAnnealingWarmUpRestarts

from attack.AttackLoss import *

def set_seed(seed=706):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_model(model_name, n_class, imgc=3, pretrained=False):
    if model_name=="resnet18":
        return ResNet18(n_class)
    elif model_name=="preresnet18":
        return PreActResNet18(n_class)
    elif model_name=="resnet34":
        return ResNet34(n_class)
    elif model_name=="resnet50":
        return ResNet50(n_class)
    elif model_name=="resnet101":
        return ResNet101(n_class)
    elif model_name=='wrn_28_10':
        return WideResNet_28_10(n_class, dropout=0.3)
    elif model_name=='wrn_34_10':
        return WideResNet_34_10(n_class, dropout=0.3)
    else:
        raise ValueError('Undefined Model Architecture')

def set_optim(model, sche, lr, epoch):
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0002)

    if sche == 'lambda98':
        lamb = 0.98
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda epoch: lamb**epoch)
    elif sche == 'lambda95':
        lamb = 0.95
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda epoch: lamb**epoch)
    elif sche == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(epoch))
    elif sche == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=int(epoch*0.2), gamma=0.5)
    elif sche == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[int(epoch*0.5), int(epoch*0.8)], gamma=0.1) 
    elif sche == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr, total_steps=epoch)
    elif sche == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=0., max_lr=lr, step_size_up=int(epoch/8), mode='triangular2') #TODO:
    elif sche == 'sdgr':
        optim = torch.optim.SGD(model.parameters(), lr=0, momentum=0.9, weight_decay=0.0002)
        scheduler = CosineAnnealingWarmUpRestarts(optim, T_0=int(epoch//3), T_mult=1, eta_max=lr,  T_up=int(epoch//15), gamma=0.5)
    elif sche == 'no':
        lamb = 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optim, lr_lambda=lambda epoch: lamb**epoch)
    else:
        raise ValueError("Wrong Learning rate Scheduler")
    
    return optim, scheduler


def set_Loss(loss_name, attack_name, model, eps, device, args):
    if loss_name == 'CE':
        return CrossEntropy(attack_name, model, eps, device, args)
    elif loss_name == 'TRADES':
        return TRADES(attack_name, model, eps, device, args)
    elif loss_name == 'QUB':
        return QUB(attack_name, model, eps, device, args)
    else:
        raise ValueError("wrong type Loss")
