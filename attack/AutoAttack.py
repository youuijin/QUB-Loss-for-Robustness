from attack.AttackBase import Attack
import torch

from autoattack import AutoAttack as auto_attack

class AutoAttack(Attack):
    def __init__(self, model, eps, args):
        self.model = model
        self.bs = args.batch_size
        self.device = torch.device(f'cuda:{args.device_num}') if args.device_num>=0 else 'cpu'

        if args.auto_version == 'custom':
            auto_list = []
            auto_attacks = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            for i in range(4):
                if args.auto_custom[i]=='1':
                    auto_list.append(auto_attacks[i])
            if len(auto_list)==0:
                raise ValueError("auto-custom mode must has at least 1 attack")
            self.at = auto_attack(self.model, norm=args.auto_norm, eps=eps/255., version=args.auto_version, attacks_to_run=auto_list, device=self.device)
        else:
            self.at = auto_attack(self.model, norm=args.auto_norm, eps=eps/255., version=args.auto_version, device=self.device)
    
    def perturb(self, x, y):
        adv_x = self.at.run_standard_evaluation(x, y)
        return adv_x

