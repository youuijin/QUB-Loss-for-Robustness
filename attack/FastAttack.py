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

        output = self.model(x + delta)
        loss = F.cross_entropy(output, y, reduction='mean')
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
    def __init__(self, model, eps, a1, a2, initial, c, device):
        self.model = model
        self.eps = eps/255.
        self.a1 = a1/255.
        self.a2 = a2/255.
        self.c = c # number of checkpoints
        self.device = device

        if initial == 'none':
            self.get_dist = self.get_standard_fgsm
        elif initial == 'uniform':
            self.get_dist = self.get_uniform_distribution

    def perturb(self, x, y):
        batch_size = x.shape[0]

        delta = torch.zeros_like(x).to(self.device)
        delta = self.get_dist(delta)
        delta.data = torch.clamp(delta, 0 - x, 1 - x)

        delta.requires_grad = True

        output = self.model(x + delta)
        loss = F.cross_entropy(output, y, reduction='mean')
        loss.backward()

        grad = delta.grad.detach()
        delta = torch.clamp(delta + self.a2* torch.sign(grad), -self.eps, self.eps) # TODO: delta.data = vs. delta = 

        delta = torch.clamp(delta, 0 - x, 1 - x)
        delta = delta.detach()

        advx = x + delta

        # Get correctly classified indexes.
        logit_clean = self.model(x)
        _, pre_clean = torch.max(logit_clean.data, 1)
        correct = (pre_clean == y)
        correct_idx = torch.masked_select(torch.arange(batch_size).to(self.device), correct)
        wrong_idx = torch.masked_select(torch.arange(batch_size).to(self.device), ~correct)
        
        # Use misclassified images as final images.
        advx[wrong_idx] = x[wrong_idx].detach()

        # Make checkpoints.
        # e.g., (batch_size*(c-1))*3*32*32 for CIFAR10.
        Xs = (torch.cat([x]*(self.c-1)) + \
              torch.cat([torch.arange(1, self.c).to(self.device).view(-1, 1)]*batch_size, dim=1).view(-1, 1, 1, 1) * \
              torch.cat([delta/self.c]*(self.c-1)))
        Ys = torch.cat([y]*(self.c-1))
                
        # Inference checkpoints for correct images.
        idx = correct_idx
        idxs = []
        self.model.eval()
        with torch.no_grad(): 
            for k in range(self.c-1):
                # Stop iterations if all checkpoints are correctly classiffied.
                if len(idx) == 0:
                    break
                # Stack checkpoints for inference.
                elif (batch_size >= (len(idxs)+1)*len(idx)):
                    idxs.append(idx + k*batch_size)
                else:
                    pass
                
                # Do inference.
                if (batch_size < (len(idxs)+1)*len(idx)) or (k==self.c-2):
                    # Inference selected checkpoints.
                    idxs = torch.cat(idxs).to(self.device)
                    pre = self.model(Xs[idxs]).detach()
                    _, pre = torch.max(pre.data, 1)
                    correct = (pre == Ys[idxs]).view(-1, len(idx))
                    
                    # Get index of misclassified images for selected checkpoints.
                    max_idx = idxs.max() + 1
                    wrong_idxs = (idxs.view(-1, len(idx))*(1-correct*1)) + (max_idx*(correct*1))
                    wrong_idx, _ = wrong_idxs.min(dim=0)
                    
                    wrong_idx = torch.masked_select(wrong_idx, wrong_idx < max_idx)
                    update_idx = wrong_idx%batch_size
                    advx[update_idx] = Xs[wrong_idx]
                    
                    # Set new indexes by eliminating updated indexes.
                    idx = torch.tensor(list(set(idx.cpu().data.numpy().tolist())\
                                            -set(update_idx.cpu().data.numpy().tolist())))
                    idxs = []

        self.model.train()
        return advx.detach()

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

class NuAT(Attack):
    '''
    # TODO:
    '''
    def __init__(self, model, eps, a1, a2, nuc_reg, device):
        self.model = model
        self.eps = eps/255.
        self.a1 = a1/255.
        self.a2 = a2/255.
        self.device = device
        self.nuc_reg = nuc_reg
        self.get_dist = self.get_bernoulli_distribution

    def perturb(self, x, y):
        delta = torch.zeros_like(x).to(self.device)
        delta = self.get_dist(delta)
        delta.data = torch.clamp(delta, 0 - x, 1 - x)

        delta.requires_grad = True

        self.model.eval()
        output = self.model(x)
        adv_output = self.model(x + delta)
        
        loss = F.cross_entropy(output ,y) + self.nuc_reg*torch.norm(output - adv_output, 'nuc')/x.shape[0]
        loss.backward()

        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + self.a2* torch.sign(grad), -self.eps, self.eps)

        delta = torch.clamp(delta, 0 - x, 1 - x)
        delta = delta.detach()

        self.model.train()

        return x + delta # advx
