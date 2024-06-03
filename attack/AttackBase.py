from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import time

class Attack(ABC):
    @abstractmethod
    def perturb(self):
        pass

    def calc_loss_acc(self, x, y):
        attack_time_st = time.time()
        adv_x = self.perturb(x, y)
        adv_logit = self.model(adv_x)
        adv_loss = F.cross_entropy(adv_logit, y)
        attack_time = (time.time() - attack_time_st)

        adv_pred = F.softmax(adv_logit, dim=1)
        adv_outputs = torch.argmax(adv_pred, dim=1)
        adv_correct_count = (adv_outputs == y).sum().item()

        return adv_loss, adv_correct_count, attack_time
    
    def get_standard_fgsm(self, a):
        return a

    def get_uniform_distribution(self, a):
        # FGSM-RS (Fast is better than free: Revisiting adversarial training, arXiv 2020)
        ## Initialize: eps * uniform(-1, 1)
        a = a.uniform_(-self.a1, self.a1)
        return a

    def get_bernoulli_distribution(self, a):
        # FGSM-BR (Guided adversarial attack for evaluating and enhancing adversarial defenses, NeurIPS 2020)
        ## Initialize: eps/2 * Bernoulli(-1, 1)
        a = self.a1 * a.bernoulli_(p=0.5)
        return a

    def get_normal_distribution(self, a):
        # FGSM-NR (Ensemble adversarial training: Attacks and defenses, ICLR 2018)
        ## Initialze: eps/2 * Normal(0, 1)
        a = self.a1 * a.normal_(0, 1)
        return a
