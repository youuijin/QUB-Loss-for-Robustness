from utils import train_utils
from torch.utils.tensorboard import SummaryWriter
from utils.SGDR import CosineAnnealingWarmUpRestarts


# for sche in ['multistep', 'lambda98', 'lambda95', 'cosine', 'step', 'onecycle', 'cyclic', 'no']:


## set seed ##
epoch = 150
lr=0

## model setting ##
model = train_utils.set_model('resnet18', 10, 3)
optim, _ = train_utils.set_optim(model, 'no', lr, epoch)
scheduler = CosineAnnealingWarmUpRestarts(optim, T_0=epoch//2, T_mult=1, eta_max=0.1,  T_up=5, gamma=0.1)
writer = SummaryWriter(f"./lrs/SDGR")

for e in range(epoch):
    scheduler.step()
    writer.add_scalar('lr', optim.param_groups[0]['lr'], e)
    