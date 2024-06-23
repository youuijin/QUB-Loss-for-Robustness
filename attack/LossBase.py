from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import time

from utils.attack_utils import set_attack

class Loss(ABC):
    def __init__(self, attack_name, model, eps, args):
        self.model = model
        if attack_name == "":
            self.attack = None
        else:
            self.attack = set_attack(attack_name, model, eps, args)
        
    @abstractmethod
    def calc_loss(self, x, y):
        #TODO:
        pass

    def calc_acc(self, x, y):
        self.model.eval()
        with torch.no_grad():
            logit = self.model(x)
            pred = F.softmax(logit, dim=1)
            outputs = torch.argmax(pred, dim=1)
            corr = (outputs == y).sum().item()

        self.model.train()

        return corr