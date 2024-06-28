import torch
import torch.nn.functional as F
from attack.AttackBase import Attack

class PGDAttack(Attack):
    def __init__(self, model, norm='Linf', eps=8.0, alpha=2.0, iter=10, restart=1, loss='CE', device=None):
        self.model = model
        self.norm = norm
        self.eps = eps/255.
        self.alpha = alpha/255.
        self.iter = iter
        self.restart = restart
        self.loss = loss
        self.device = device

    def perturb(self, x_natural, y):
        max_loss = torch.zeros(y.shape[0]).to(self.device)
        max_x = torch.zeros_like(x_natural).to(self.device)
        
        for _ in range(self.restart):
            x = x_natural.detach()
            x = x + torch.zeros_like(x).uniform_(-self.eps, self.eps)
            for _ in range(self.iter):
                x.requires_grad_()
                with torch.enable_grad():
                    logits = self.model(x)
                    if self.loss == 'CE':
                        loss = F.cross_entropy(logits, y)
                    else:
                        criterion_kl = torch.nn.KLDivLoss(reduction='sum')
                        loss = criterion_kl(F.log_softmax(self.model(x), dim=1), F.softmax(self.model(x_natural), dim=1))
                grad = torch.autograd.grad(loss, [x])[0]
                x = x.detach() + self.alpha * torch.sign(grad.detach())
                x = torch.min(torch.max(x, x_natural - self.eps), x_natural + self.eps)
                x = torch.clamp(x, 0, 1)
            
            with torch.no_grad():
                # all_loss = self.loss_func(self.model(x), y, reduction='none')
                if self.loss == 'CE':
                    all_loss = F.cross_entropy(self.model(x), y, reduction='none')
                else:
                    criterion_kl = torch.nn.KLDivLoss(reduction='none')
                    all_loss = criterion_kl(F.log_softmax(self.model(x), dim=1), 
                                            F.softmax(self.model(x_natural), dim=1))
        
                    all_loss = all_loss.sum(dim=1)
                max_x[all_loss >= max_loss] = torch.clone(x.detach()[all_loss >= max_loss])
                max_loss = torch.max(max_loss, all_loss)

        return max_x
