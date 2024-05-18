from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import time

class Attack(ABC):
    @abstractmethod
    def perturb(self):
        pass

    def calc_loss_acc(self, x, y):
        lipschitz = 0.5
        attack_time_st = time.time()
        adv_x = self.perturb(x, y)
        attack_time = (time.time() - attack_time_st)

        adv_logit = self.model(adv_x)
        adv_loss = F.cross_entropy(adv_logit, y)
        adv_pred = F.softmax(adv_logit, dim=1)
        adv_outputs = torch.argmax(adv_pred, dim=1)
        adv_correct_count = (adv_outputs == y).sum().item()

        with torch.no_grad():
            logit = self.model(x)
            softmax = F.softmax(logit, dim=1)
            y_onehot = F.one_hot(y, num_classes = softmax.shape[1])
            approx_metric = torch.abs(torch.norm(adv_logit - (logit - 1.0/lipschitz*(softmax-y_onehot)), dim=1))
            approx_metric = approx_metric.mean().item()/torch.abs(logit).mean().item()

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
    
    def get_flat_distribution(self, x):
        delta = torch.zeros_like(x).to(x.device)

        delta.requires_grad = True
        output = self.model(x + delta)
        pred = F.softmax(output, dim=1)
        loss = -1 * (1 - torch.sum(pred**2, dim=1)).mean()
        loss.backward()

        grad = delta.grad.detach()
        delta.data = torch.clamp(delta - self.a1 * torch.sign(grad), -self.eps, self.eps)
        # TODO: different step size, not only sign

        delta = delta.detach()

        return delta
    
    def get_flat_grad_distribution(self, x):
        delta = torch.zeros_like(x).to(x.device)

        delta.requires_grad = True
        output = self.model(x + delta)
        pred = F.softmax(output, dim=1)
        loss = -1 * (1 - torch.sum(pred**2, dim=1)).mean()
        loss.backward()

        grad = delta.grad.detach()
        max_grads = torch.max(grad.view(x.shape[0], -1), dim=1).values
        max_grads = torch.where(max_grads == 0.0, torch.tensor(1.0), max_grads).to(x.device)
        scaled_grad = grad / max_grads.view(-1, 1, 1, 1)

        delta.data = torch.clamp(delta - self.a1 * scaled_grad, -self.eps, self.eps)

        delta = delta.detach()
        return delta
