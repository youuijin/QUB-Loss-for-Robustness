from attack.AttackBase import Attack
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from utils.model.FGSM_SDI_attacker import One_Layer_Attacker

class FGSM(Attack):
    def __init__(self, model, eps, a1, a2, initial, device):
        self.model = model
        self.eps = eps/255.
        self.a1 = a1/255.
        self.a2 = a2/255.
        self.device = device

        if initial == 'none':
            self.get_dist = self.get_standard_fgsm
        elif initial == 'uniform':
            self.get_dist = self.get_uniform_distribution
        elif initial == 'bernoulli':
            self.get_dist = self.get_bernoulli_distribution
        elif initial == 'normal':
            self.get_dist = self.get_normal_distribution

    def perturb(self, x, y):
        delta = torch.zeros_like(x).to(self.device)
        delta = self.get_dist(delta)
        delta.data = torch.clamp(delta, 0 - x, 1 - x)

        delta.requires_grad = True

        # TODO: Change to CE Loss after testing QQ experiments
        logit = self.model(x)
        adv_logit = self.model(x + delta)
        adv_norm = torch.norm(adv_logit-logit, dim=1)
        softmax = F.softmax(logit, dim=1)
        y_onehot = F.one_hot(y, num_classes = softmax.shape[1])
        loss = F.cross_entropy(logit, y, reduction='none')
        loss_adv = loss + torch.sum((adv_logit-logit)*(softmax-y_onehot), dim=1) + 0.5/2.0*torch.pow(adv_norm, 2)
        loss = loss_adv.mean()

        # output = self.model(x + delta)
        # loss = F.cross_entropy(output, y, reduction='mean')
        loss.backward()

        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + self.a2* torch.sign(grad), -self.eps, self.eps)

        delta = torch.clamp(delta, 0 - x, 1 - x)
        delta = delta.detach()

        return x + delta # advx

class Free(Attack):
    '''
    Adversarial Training for Free!
    Shafahi et al., NeurIPS 2019
    '''
    pass

class YOPO(Attack):
    '''
    You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle
    Zhang et al,. NeurIPS 2019
    '''
    pass

class FGSM_GA(Attack):
    '''
    Understanding and improving fast adversarial training
    Andriushchenko et al., NeurIPS 2020
    '''
    pass

class FGSM_CKPT(Attack):
    '''
    Understanding catastrophic overfitting in single-step adversarial training
    Kim et al., AAAI 2021
    '''
    pass

class FGSM_SDI(Attack):
    '''
    Boosting Fast Adversarial Training with Learnable Adversarial Initializaiton
    Jia et al., TIP 2022
    '''
    def __init__(self, model, eps, alpha, lr_att, device, args):
        self.model = model
        self.eps = eps/255.
        self.alpha = alpha/255.
        self.device = device
        self.attacker = One_Layer_Attacker(eps=self.eps, input_channel=6).to(self.device)
        self.optimizer_att = torch.optim.SGD(self.attacker.parameters(), lr=lr_att, momentum=0.9, weight_decay=5e-4)
        self.n_way = args.n_way
        self.step = 0
        self.att_num = 20 # from paper
        tot_steps = args.steps * args.epoch
        self.attacker_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_att,
                                                      milestones=[int(tot_steps * 0.5), int(tot_steps * 0.8)], gamma=0.1)

    def perturb(self, x, y):
        x.requires_grad_()
        loss = F.cross_entropy(self.model(x), y)
        grad = torch.autograd.grad(loss, [x])[0]

        # update attacker
        if self.step%self.att_num == 0:
            self.optimizer_att.zero_grad()
            self.attacker.zero_grad()
            self.model.zero_grad()
            loss = self.perturb_one_time(x, y, grad, update=1)
            loss.backward()
            self.optimizer_att.step()
        
        self.model.zero_grad()
        self.attacker.zero_grad()
        advx = self.perturb_one_time(x, y, grad, update=0)

        self.attacker_scheduler.step()
        self.step += 1

        return advx

    def perturb_one_time(self, x, y, grad, update):
        if update == 0:
            self.model.train()
            self.attacker.eval()
        else:
            self.attacker.train()

        ## optional : Label Smoothing

        adv_input = torch.cat([x, 1.0*(torch.sign(grad))], 1).detach()
        init_perturb = self.attacker(adv_input)
        advx = x + init_perturb
        advx = torch.clamp(advx, 0, 1)
        
        advx.requires_grad_()

        with torch.enable_grad():
            loss_adv = F.cross_entropy(self.model(advx), y)
            grad_adv = torch.autograd.grad(loss_adv, [advx])[0]
            perturbation = torch.clamp(self.alpha* torch.sign(grad_adv), -self.eps, self.eps)

        perturbation = torch.clamp(init_perturb + perturbation, -self.eps, self.eps)

        advx = x + perturbation
        advx = torch.clamp(advx, 0, 1)

        if update == 1:
            logit = self.model(advx)
            one_hot = np.eye(self.n_way)[y.to(self.device).data.cpu().numpy()]
            result = one_hot * 0.5 + (one_hot - 1.) * ((0.5 - 1) / float(10 - 1))
            label_smoothing = Variable(torch.tensor(result).to(self.device))
            log_prob = F.log_softmax(logit, dim=-1)
            loss = (-label_smoothing.float() * log_prob).sum(dim=-1).mean()
            loss_adv = -loss

            return loss_adv

        return advx

        