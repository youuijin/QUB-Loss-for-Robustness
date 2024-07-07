import torch, csv, os
import torch.nn.functional as F

from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
from torch.utils.data import DataLoader

# from attack.PGD import PGDAttack
from attack.AutoAttack import AutoAttack
from utils import attack_utils
from attack import IterativeAttack

class Tester:
    def __init__(self, args, model, device, log_name):
        self.device = device
        self.model = model.to(device)
        self.save_path = f'./results/{args.dataset}/{args.model}'
        self.args = args

        self.test_eps = args.test_eps
        self.test_method = ['AA', 'PGD']
        self.dataset = args.dataset

        if log_name is not None:
            # test after train, don't neet to check csv files
            log_name = log_name.replace("/", "_")
            self.model_paths = [f'{args.model}_{log_name}.pt', f'{args.model}_{log_name}_last.pt']
        else:
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

    def set_tested_model(self):
        self.tested_model_paths = []
        f = open(f'{self.save_path}/csvs/test_result.csv', 'r', encoding='utf-8')
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

    def attack_method(self, test_name):
        if test_name == "AA":
            self.test_at = AutoAttack(self.model, eps=self.test_eps, args=self.args)
        else:
            self.test_at = IterativeAttack.PGDAttack(self.model, eps=self.test_eps, alpha=2., iter=50, restart=10, loss='CE', device=self.device)
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
            accuracies = []

            print('Testing Start')
            print(f'Model Name : {model_path}, test_method : {self.test_method}')
            # clean accuracy
            test_correct_count = 0
            for _, (x, y) in enumerate(tqdm(self.test_loader, desc=f'clean test')):
                x, y = x.to(self.device), y.to(self.device)
                correct_count = self.inference(x, y)
                test_correct_count += correct_count
            accuracies.append(round(test_correct_count/len(self.test_data)*100, 4))


           

            # robust accuracy
            for test_name in self.test_method:
                self.attack_method(test_name)

                test_adv_correct_count = 0
                for _, (x, y) in enumerate(tqdm(self.test_loader, desc=f'{test_name} test')):
                    x, y = x.to(self.device), y.to(self.device)
                    x_adv = self.test_at.perturb(x, y)
                    adv_correct_count = self.inference(x_adv, y)
                    test_adv_correct_count += adv_correct_count

                test_adv_acc = round(test_adv_correct_count/len(self.test_data)*100, 4)
                accuracies.append(test_adv_acc)

            result = [model_path, self.test_eps] + accuracies
            with open(f'{self.save_path}/csvs/test_result.csv', 'a', encoding='utf-8', newline='') as f:
                wr = csv.writer(f)
                wr.writerow(result)

    def inference(self, x, y):
        logit = self.model(x)
        pred = F.softmax(logit, dim=1)
        outputs = torch.argmax(pred, dim=1)
        correct_count = (outputs == y).sum().item()

        return correct_count