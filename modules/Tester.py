import torch, csv, os
import torch.nn.functional as F

from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torch.utils.data import DataLoader

from attack.PGD import PGDAttack
from attack.Auto import AutoAttack

class Tester:
    def __init__(self, args, model, device):
        self.device = device
        self.model = model.to(device)
        self.save_path = f'./results/{args.dataset}'
        self.args = args

        self.test_eps = args.test_eps
        self.test_method = args.test_method
        self.dataset = args.dataset

        # auto attack
        if args.test_method == "AA":
            self.csv = 'AutoAttack'
        elif args.test_method == "AA_ckpt":
            self.csv = 'AutoAttack_ckpt'
        # adversarial attack - PGD_Linf
        else:
            self.csv = 'PGD_Attack'

        self.set_tested_model()

        if args.model_path != "":
            if self.check_is_tested(args.model_path+".pt"):
                raise ValueError("This model is already tested")
            self.model_paths = [f'{args.model_path}.pt']
        else:
            exist_model_paths = os.listdir(f'{self.save_path}/saved_model')
            self.model_paths = []
            
            for exist_model_path in exist_model_paths:
                if not self.check_is_tested(exist_model_path):
                    self.model_paths.append(exist_model_path)

        self.set_dataset(args)
        self.attack_method()

    def set_tested_model(self):
        self.tested_model_paths = []
        f = open(f'{self.save_path}/csvs/{self.csv}.csv', 'r', encoding='utf-8')
        rdr = csv.reader(f)
        next(rdr)
        for line in rdr:
            self.tested_model_paths.append(line[0].strip().replace("/", "_"))
        f.close() 

    def check_is_tested(self, model_path):
        if model_path in self.tested_model_paths:
            return True
        else:
            return False

    def attack_method(self):
        if "AA" in self.test_method:
            self.test_at = AutoAttack(self.model, eps=self.test_eps, args=self.args)
        else:
            self.test_at = PGDAttack(self.model, eps=self.test_eps, alpha=2., iter=50, restart=10, norm='Linf')

    def set_dataset(self, args):
        ### normalize setting ###
        if args.normalize == "imagenet":
            norm_mean, self.norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        elif args.normalize == "twice":
            norm_mean, self.norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        else: 
            norm_mean, self.norm_std = (0, 0, 0), (1, 1, 1)

        ### dataset ###
        transform = transforms.Compose([transforms.CenterCrop((args.imgsz, args.imgsz)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(norm_mean, self.norm_std)])
        
        if args.dataset == 'cifar10':
            self.test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)
        elif args.dataset == 'cifar100':
            self.test_data = CIFAR100(root='./data', train=False, download=True, transform=transform)
        elif args.dataset == 'svhn':
            self.test_data = SVHN(root='./data', split='test', download=True, transform=transform)
        self.test_loader = DataLoader(self.test_data, batch_size=args.batch_size, shuffle=True, num_workers=0)


    def test(self):
        for model_path in self.model_paths:
            if model_path[-2:] != 'pt':
                continue
            self.model.load_state_dict(torch.load(f'{self.save_path}/saved_model/{model_path}', map_location=self.device))
            self.model.eval()

            test_correct_count = 0
            test_adv_correct_count = 0
            for _, (x, y) in enumerate(tqdm(self.test_loader, desc=f'{self.test_method} test')):
                x = x.to(self.device)
                y = y.to(self.device)
                x_adv = self.test_at.perturb(x, y)
                correct_count = self.inference(x, y)
                adv_correct_count = self.inference(x_adv, y)
                test_correct_count += correct_count
                test_adv_correct_count += adv_correct_count
            
            test_acc =  round(test_correct_count/len(self.test_data)*100, 4)
            test_adv_acc = round(test_adv_correct_count/len(self.test_data)*100, 4)

            result = [model_path, self.test_eps, test_acc, test_adv_acc]
            with open(f'{self.save_path}/csvs/{self.csv}.csv', 'a', encoding='utf-8', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(result)

    def inference(self, x, y):
        logit = self.model(x)
        pred = F.softmax(logit, dim=1)
        outputs = torch.argmax(pred, dim=1)
        correct_count = (outputs == y).sum().item()

        return correct_count