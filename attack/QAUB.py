import torch.nn.functional as F
import torch
from attack.AttackBase import Attack
from attack.FGSM import FGSM
from attack.PGD import PGDAttack
from attack.rLF import rLFAttack
import time

class QAUB(Attack):
    def __init__(self, model, eps, step, lipschitz):
        self.model = model
        self.step = step
        self.lipschitz = lipschitz
        self.eps = eps

        if self.step==-1:
            self.loss_func = self.step1_pgd
        elif self.step==0:
            self.loss_func = self.step1_fgsm
        elif self.step==1:
            self.loss_func = self.step1_rLF
        elif self.step==2:
            self.loss_func = self.step2
        elif self.step==3:
            self.loss_func = self.step3
        elif self.step==4:
            self.loss_func = self.step4
        elif self.step==5:
            self.loss_func = self.step5
        elif self.step==6:
            self.loss_func = self.step6
        # elif self.step==7:
        #     self.loss_func = self.step1_pgd
        # elif self.step==8:
        #     self.loss_func = self.step8
        # elif self.step==10:
        #     self.loss_func = self.step10
        # elif self.step==11:
        #     self.loss_func = self.step11
        # elif self.step==12:
        #     self.loss_func = self.step12
        # elif self.step==13:
        #     self.loss_func = self.step13
        # elif self.step==14:
        #     self.loss_func = self.step14

    def perturb(self, x, y):
        return None

    def calc_loss_acc(self, x, y):
        attack_time_st = time.time()
        approx_loss, adv_loss = self.loss_func(x, y)
        attack_time = (time.time() - attack_time_st)
        return approx_loss, adv_loss, attack_time
    
    def step1_fgsm(self, x, y):
        # Quadratic upper bound
        # L(h(x')) <= L(h(x))+(h(x')-h(x))L'(h(x)) + K/2*||h(x')-h(x)||^2

        logit = self.model(x)
        softmax = F.softmax(logit, dim=1)
        y_onehot = F.one_hot(y, num_classes = softmax.shape[1])
        
        # fgsm = FGSM(self.model, self.eps, 0, self.eps, initial='none') ## original FGSM
        fgsm = FGSM(self.model, self.eps, self.eps/2, self.eps/2, initial='uniform') ## FGSM-RS
        f_x = fgsm.perturb(x, y)
        f_logit = self.model(f_x)
        f_norm = torch.norm(f_logit-logit, dim=1)

        loss = F.cross_entropy(logit, y, reduction='none')
        adv_loss = F.cross_entropy(f_logit, y)

        f_approx_loss = loss + torch.sum((f_logit-logit)*(softmax-y_onehot), dim=1) + self.lipschitz/2.0*torch.pow(f_norm, 2)
        
        # bound_rate = (f_adv_loss<=f_approx_loss).sum().item() 
        return f_approx_loss.mean(), adv_loss
    
    def step1_rLF(self, x, y):
        # Quadratic upper bound
        # L(h(x')) <= L(h(x))+(h(x')-h(x))L'(h(x)) + K/2*||h(x')-h(x)||^2
        logit = self.model(x)
        softmax = F.softmax(logit, dim=1)
        y_onehot = F.one_hot(y, num_classes = softmax.shape[1])

        rlf = rLFAttack(self.model, self.eps, self.eps/2, self.eps/2, initial='uniform') ## rLF-RS
        lf_x = rlf.perturb(x, y)
        lf_logit = self.model(lf_x)
        lf_norm = torch.norm(lf_logit-logit, dim=1)

        loss = F.cross_entropy(logit, y, reduction='none')
        adv_loss = F.cross_entropy(lf_logit, y)

        lf_approx_loss = loss + torch.sum((lf_logit-logit)*(softmax-y_onehot), dim=1) + self.lipschitz/2.0*torch.pow(lf_norm, 2)

        return lf_approx_loss.mean(), adv_loss
    
    def step1_pgd(self, x, y):
        # Quadratic upper bound
        # L(h(x')) <= L(h(x))+(h(x')-h(x))L'(h(x)) + K/2*||h(x')-h(x)||^2

        logit = self.model(x)
        softmax = F.softmax(logit, dim=1)
        y_onehot = F.one_hot(y, num_classes = softmax.shape[1])
        
        pgd = PGDAttack(self.model)
        p_x = pgd.perturb(x, y)
        p_logit = self.model(p_x)
        p_norm = torch.norm(p_logit-logit, dim=1)

        loss = F.cross_entropy(logit, y, reduction='none')
        adv_loss = F.cross_entropy(p_logit, y)

        p_approx_loss = loss + torch.sum((p_logit-logit)*(softmax-y_onehot), dim=1) + self.lipschitz/2.0*torch.pow(p_norm, 2)
        
        # bound_rate = (f_adv_loss<=f_approx_loss).sum().item() 
        return p_approx_loss.mean(), adv_loss

    def step2(self, x, y):
        # fgsm = FGSM(self.model, self.eps, 0, self.eps, initial='none') ## FGSM
        fgsm = FGSM(self.model, self.eps, self.eps/2, self.eps/2, initial='uniform') ## FGSM-RS
        f_x = fgsm.perturb(x, y)

        # with torch.no_grad():
        logit = self.model(x)
        logit_adv = self.model(f_x)
        # logit_for_ce = self.model(x)
        loss = F.cross_entropy(logit, y)

        softmax = F.softmax(logit, dim=1)
        y_onehot = F.one_hot(y, num_classes = softmax.shape[1])

        # cross entropy 미분 -> y'-y
        approx_loss = torch.pow(torch.norm(logit_adv - (logit - 1.0/self.lipschitz*(softmax-y_onehot)), dim=1),2)

        # print("ce", loss.item())
        # print("qaub", approx_loss.mean().item())

        return approx_loss.mean()

    def step3(self, x, y):
        # approximation #1-1
        # linear approximation
        # ||delta*h'(x)+1/K*L'(h(x))||^2
        x = x.requires_grad_(True)

        # fgsm = FGSM(self.model, self.eps, 0, self.eps, initial='none')
        fgsm = FGSM(self.model, self.eps, self.eps/2, self.eps/2, initial='uniform')
        f_x = fgsm.perturb(x, y)
        ##################################################
        # delta를 정확히 계산
        delta = (f_x - x)
        # delta를 그냥 eps로 계산?

        ##################################################

        logit = self.model(x)
        softmax = F.softmax(logit, dim=1)
        y_onehot = F.one_hot(y, num_classes = softmax.shape[1])

        # calculate jacobian
        x_copy = x.detach().clone().requires_grad_(True)
        logits_copy = self.model(x_copy)
        logit_sum = logits_copy.sum(dim=0)

        b, c, h, w = x_copy.size()
        jacobian = torch.zeros(b, c, h, w, logit_sum.shape[0])
        for i in range(len(logit_sum)):
            logit_sum[i].backward(retain_graph=True)
            jacobian[:, :, :, :, i] = x_copy.grad
            x_copy.grad.zero_()
        
        delta = torch.reshape(delta, (delta.shape[0], -1)).unsqueeze(1)
        jacobian = torch.reshape(jacobian, (jacobian.shape[0], -1, jacobian.shape[-1])).to(x.device)

        # err = torch.norm(self.model(f_x)+torch.matmul(delta, jacobian).squeeze()-logit)

        # cross entropy 미분 -> y'-y
        approx_loss = torch.pow(torch.norm(torch.matmul(delta, jacobian).squeeze()+1.0/self.lipschitz*(softmax-y_onehot), dim=1), 2)
        # print(approx_loss.mean())
        loss = F.cross_entropy(logit, y)
        return loss + approx_loss.mean()

    def step4(self, x, y):
        # linear approximation + heuristic
        logit = self.model(x)
        pred = F.softmax(logit, dim=1)
        y_onehot = F.one_hot(y, num_classes = pred.shape[1])
        approx_loss = torch.pow((1.0+self.eps/255.)/self.lipschitz*torch.norm(pred-y_onehot, dim=1), 2)

        # approx_loss = torch.pow(self.train_eps*torch.norm(pred-y_onehot), 2)
        loss = F.cross_entropy(logit, y)
        return loss + approx_loss.mean()
    
    def step5(self, x, y):
        # plus upper bound
        logit = self.model(x)
        pred = F.softmax(logit, dim=1)
        loss = F.cross_entropy(logit, y)
        y_onehot = F.one_hot(y, num_classes = pred.shape[1])
        approx_loss = loss - 1.0/(2*self.lipschitz)*torch.pow(torch.norm(pred-y_onehot, dim=1),2)
        # approx_loss = loss + torch.sum(logit*(pred-y_onehot), dim=1) - 1/(2.0 * self.lipschitz)*torch.pow(torch.norm(pred-y_onehot, dim=1), 2)

        # loss = F.cross_entropy(logit, y)
        return loss + approx_loss.mean()
    
    def step6(self, x, y):
        # plus upper bound
        logit = self.model(x)
        pred = F.softmax(logit, dim=1)
        loss = F.cross_entropy(logit, y)
        y_onehot = F.one_hot(y, num_classes = pred.shape[1])
        approx_loss = loss + 3.0/(2*self.lipschitz)*torch.pow(torch.norm(pred-y_onehot, dim=1),2)
        # return approx_loss.mean()
        loss = F.cross_entropy(logit, y)
        return loss + approx_loss.mean()

    # def step7(self, x, y):
    #     # Quadratic upper bound
    #     # L(h(x')) <= L(h(x))+(h(x')-h(x))L'(h(x)) + K/2*||h(x')-h(x)||^2

    #     logit = self.model(x)
    #     softmax = F.softmax(logit, dim=1)
    #     y_onehot = F.one_hot(y, num_classes = softmax.shape[1])
        
    #     fgsm = FGSM(self.model, self.eps, 0, self.eps, initial='none')
    #     f_x = fgsm.perturb(x, y)
    #     f_logit = self.model(f_x)
    #     # f_norm = torch.norm(f_logit-logit, dim=1)
        
    #     approx_norm = torch.norm(F.softmax(approx_logit, dim=1) - F.softmax(logit, dim=1), dim=1)

    #     loss = F.cross_entropy(logit, y, reduction='none')

    #     f_adv_loss = F.cross_entropy(f_logit, y, reduction='none')
    #     f_approx_loss = loss + torch.sum((approx_logit-logit)*(softmax-y_onehot), dim=1) + self.lipschitz/2.0*torch.pow(approx_norm, 2)

    #     # print(loss.mean().item(), torch.sum((approx_logit-logit)*(softmax-y_onehot), dim=1).mean().item(),self.lipschitz/2.0*torch.pow(approx_norm, 2).mean().item() )

    #     bound_rate = (f_adv_loss<=f_approx_loss).sum().item() 

    #     return f_approx_loss.mean(), f_adv_loss.mean(), bound_rate
    
    # def step8(self, x, y):
    #     # plus upper bound (Hessian)
    #     logit = self.model(x)
    #     pred = F.softmax(logit, dim=1)
    #     hessian = - torch.bmm(pred.unsqueeze(-1), pred.unsqueeze(-1).transpose(-2, -1)) + torch.diag_embed(pred, dim1=-2, dim2=-1)

    #     eigenvalues, _ = torch.linalg.eig(hessian)
    #     approx_loss, _ = torch.max(eigenvalues.real, dim=1)

    #     return approx_loss.mean()
    
    # def step9(self, x, y):
    #     logit = self.model(x)
    #     pred = F.softmax(logit, dim=1)
    #     loss = F.cross_entropy(logit, y)
        
    #     y_onehot = F.one_hot(y, num_classes = pred.shape[1])
    #     approx_loss_4 = torch.pow((1.0+self.train_eps)/self.lipschitz*torch.norm(pred-y_onehot, dim=1), 2)
    #     approx_loss_6 = loss + 3.0/(2*self.lipschitz)*torch.pow(torch.norm(pred-y_onehot, dim=1),2)
    #     approx_loss = approx_loss_4 + approx_loss_6
    #     return approx_loss.mean()

    # def step10(self, x, y):
    #     # 모든 원소의 제곱 합
    #     logit = self.model(x)
    #     pred = F.softmax(logit, dim=1)
    #     hessian = - torch.bmm(pred.unsqueeze(-1), pred.unsqueeze(-1).transpose(-2, -1)) + torch.diag_embed(pred, dim1=-2, dim2=-1)

    #     approx_loss = torch.sum(hessian**2, dim=(1,2))
    #     return approx_loss.mean()

    # def step11(self, x, y):
    #     # 1-\sum y_i^2
    #     logit = self.model(x)
    #     pred = F.softmax(logit, dim=1)
    #     approx_loss = 1 - torch.sum(pred**2, dim=1)

    #     return approx_loss.mean()
    
    # def step12(self, x, y):
    #     # 1-\sum y_i^2
    #     logit = self.model(x)
    #     pred = F.softmax(logit, dim=1)
    #     approx_loss_a = 1 - torch.sum(pred**2, dim=1) # step 11

    #     eps = torch.rand(x.shape[0]).to(x.device)*(8/255.)

    #     fgsm = attacks.GradientSignAttack(self.model, eps=eps)
    #     f_x = fgsm.perturb(x, y)
    #     f_logit = self.model(f_x)
    #     f_adv_loss = F.cross_entropy(f_logit, y, reduction='none')

    #     approx_loss = approx_loss_a + f_adv_loss
    #     # print(approx_loss_a.shape)
    #     # print(f_adv_loss.shape)
    #     # print(approx_loss.shape)
    #     # print(approx_loss_a.mean())
    #     # print(f_adv_loss.mean())
    #     # exit()
        
    #     return approx_loss.mean()
    
    
    # def step13(self, x, y):
    #     # 1-\sum y_i^2
    #     eps = torch.rand(x.shape[0]).to(x.device)*(8/255.)

    #     fgsm = attacks.GradientSignAttack(self.model, eps=eps)
    #     f_x = fgsm.perturb(x, y)
    #     f_logit = self.model(f_x)
    #     f_adv_loss = F.cross_entropy(f_logit, y, reduction='none')
        
    #     return f_adv_loss.mean()

    # def step14(self, x, y):
    #     # 1-\sum y_i^2
    #     logit = self.model(x)
    #     pred = F.softmax(logit, dim=1)
    #     approx_loss_a = 1 - torch.sum(pred**2, dim=1) # step 11


    #     fgsm = attacks.GradientSignAttack(self.model, eps=self.train_eps/255.)
    #     f_x = fgsm.perturb(x, y)
    #     f_logit = self.model(f_x)
    #     f_adv_loss = F.cross_entropy(f_logit, y, reduction='none')

    #     approx_loss = approx_loss_a + f_adv_loss
        
    #     return approx_loss.mean()attack_time = (time.time() - attack_time_st)
    