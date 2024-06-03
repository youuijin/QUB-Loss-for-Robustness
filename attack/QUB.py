import torch.nn.functional as F
import torch
from attack.AttackBase import Attack
from attack.FGSM import FGSM
from attack.PGD import PGDAttack
import time

class QUB(Attack):
    def __init__(self, model, eps, lipschitz, a1, a2):
        self.model = model
        self.lipschitz = lipschitz
        self.eps = eps

        self.attack = FGSM(self.model, self.eps, a1, a2, initial='uniform') ## FGSM-RS 
        # self.attack = PGDAttack(self.model, self.eps, iter) ## PGD-Linf

        self.loss_func = self.qub

    def perturb(self, x, y):
        return None

    
    def calc_loss_acc(self, x, y):
        attack_time_st = time.time()
        logit = self.model(x)
        softmax = F.softmax(logit, dim=1)
        y_onehot = F.one_hot(y, num_classes = softmax.shape[1])
        
        advx = self.attack.perturb(x, y)
        adv_logit = self.model(advx)
        adv_norm = torch.norm(adv_logit-logit, dim=1)

        loss = F.cross_entropy(logit, y, reduction='none')

        upper_loss = loss + torch.sum((adv_logit-logit)*(softmax-y_onehot), dim=1) + self.lipschitz/2.0*torch.pow(adv_norm, 2)

        attack_time = (time.time() - attack_time_st)
        return upper_loss.mean(), 0, attack_time
    