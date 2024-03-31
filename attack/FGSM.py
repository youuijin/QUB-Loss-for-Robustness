from attack.AttackBase import Attack
import torch
import torch.nn.functional as F

class FGSM(Attack):
    def __init__(self, model, eps, alpha=2., initial='none'):
        self.model = model
        self.eps = eps/255.
        self.alpha = alpha/255.

        if initial == 'none':
            self.get_dist = self.get_standard_fgsm
        elif initial == 'uniform':
            self.get_dist = self.get_uniform_distribution
        elif initial == 'bernoulli':
            self.get_dist = self.get_bernoulli_distribution
        elif initial == 'normal':
            self.get_dist = self.get_normal_distribution
        elif initial == 'flat':
            self.get_dist = self.get_flat_distribution

    def perturb(self, x, y):
        delta = torch.zeros_like(x).to(x.device)
        delta = self.get_dist(delta)
        delta.data = torch.clamp(delta, 0 - x, 1 - x)

        delta.requires_grad = True
        output = self.model(x + delta)
        loss = F.cross_entropy(output, y, reduction='mean')
        loss.backward()

        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + (self.eps - self.alpha)* torch.sign(grad), -self.eps, self.eps)

        delta = torch.clamp(delta, 0 - x, 1 - x)
        delta = delta.detach()

        return x + delta
    