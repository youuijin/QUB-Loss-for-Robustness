from attack.LossBase import Loss
import torch.nn.functional as F
import torch, time
from utils.attack_utils import set_attack

class CrossEntropy(Loss):
    def __init__(self, attack_name, model, eps, device, args):
        super().__init__(attack_name, model, eps, device, args)

    def calc_loss(self, x, y):
        loss_time_st = time.time()
        attack_time = 0
        if self.attack is not None:
            attack_time_st = time.time()
            advx = self.attack.perturb(x, y)
            attack_time = (time.time() - attack_time_st)
            logit = self.model(advx)

            loss = F.cross_entropy(logit, y)
        else:
            logit = self.model(x)
            loss = F.cross_entropy(logit, y)

        loss_time = time.time() - loss_time_st
        return loss, attack_time, loss_time

'''
TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)
    "Theoretically Principled Trade-off between Robustness and Accuracy", ICML '19
    Hongyang Zhang (CMU, TTIC), Yaodong Yu (University of Virginia), et al.
'''
class TRADES(Loss):
    def __init__(self, attack_name, model, eps, device, args):
        super().__init__(attack_name, model, eps, device, args)
        assert self.attack is not None
        self.attack = set_attack(attack_name, model, eps, device, args, type='KL')
        self.criterion_kl = torch.nn.KLDivLoss(reduction='sum')
        self.beta = args.beta

    def calc_loss(self, x, y):
        loss_time_st = time.time()
        logit = self.model(x)

        attack_time_st = time.time()
        advx = self.attack.perturb(x, y)
        attack_time = (time.time() - attack_time_st)

        loss = F.cross_entropy(logit, y)
        
        loss_adv = (1.0/x.shape[0]) * self.criterion_kl(F.log_softmax(self.model(advx), dim=1), F.softmax(self.model(x), dim=1)+1e-10)

        loss_time = time.time() - loss_time_st

        return loss + self.beta * loss_adv, attack_time, loss_time

'''

ours
'''
class QUB(Loss):
    def __init__(self, attack_name, model, eps, device, args):
        super().__init__(attack_name, model, eps, device, args)
        self.lipschitz = 0.5
        assert self.attack is not None
    
    def calc_loss(self, x, y):
        loss_time_st = time.time()
        logit = self.model(x)
        softmax = F.softmax(logit, dim=1)
        y_onehot = F.one_hot(y, num_classes = softmax.shape[1])
        
        attack_time_st = time.time()
        advx = self.attack.perturb(x, y)
        attack_time = (time.time() - attack_time_st)
        adv_logit = self.model(advx)
        adv_norm = torch.norm(adv_logit-logit, dim=1)

        loss = F.cross_entropy(logit, y, reduction='none')

        upper_loss = loss + torch.sum((adv_logit-logit)*(softmax-y_onehot), dim=1) + self.lipschitz/2.0*torch.pow(adv_norm, 2)

        loss_time = time.time() - loss_time_st
        return upper_loss.mean(), attack_time, loss_time
    