import torch
import torch.nn.functional as F
from attack.AttackBase import Attack

class PGDAttack(Attack):
    def __init__(self, model, norm='Linf', eps=8.0, alpha=2.0, iter=10, restart=1):
        self.model = model
        self.norm = norm
        self.eps = eps/255.
        self.alpha = alpha/255.
        self.iter = iter
        self.restart = restart

    def perturb(self, x_natural, y):
        max_loss = torch.zeros(y.shape[0]).to(x_natural.device)
        max_x = torch.zeros_like(x_natural).to(x_natural.device)
        
        for _ in range(self.restart):
            x = x_natural.detach()
            x = x + torch.zeros_like(x).uniform_(-self.eps, self.eps)
            for _ in range(self.iter):
                x.requires_grad_()
                with torch.enable_grad():
                    logits = self.model(x)
                    loss = F.cross_entropy(logits, y)
                grad = torch.autograd.grad(loss, [x])[0]
                x = x.detach() + self.alpha * torch.sign(grad.detach())
                x = torch.min(torch.max(x, x_natural - self.eps), x_natural + self.eps)
                x = torch.clamp(x, 0, 1)
            
            with torch.no_grad():
                all_loss = F.cross_entropy(self.model(x), y, reduction='none')
                max_x[all_loss >= max_loss] = torch.clone(x.detach()[all_loss >= max_loss])
                max_loss = torch.max(max_loss, all_loss)

        return max_x
