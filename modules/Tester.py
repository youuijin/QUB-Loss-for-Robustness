import torch, csv, os
import torch.nn.functional as F

from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from attack.PGD import PGDAttack
from attack.Auto import AutoAttack
from attack.FGSM import FGSM

# TODO: load_state_dict는 객체를 바꾸지 않는지 확인, 더 간단한 코드 작성하기

class Tester:
    def __init__(self, args, model, device):
        self.device = device
        self.model = model.to(device)
        self.save_path = args.save_path
        self.args = args
        self.lipschitz = args.lipschitz

        self.test_eps = args.test_eps
        self.test_method = args.test_method
        if args.test_method == "bound":
            self.csv = 'Bound'
        # auto attack
        elif args.test_method == "AA":
            self.csv = 'AutoAttack'
        elif args.test_method == "AA_ckpt":
            self.csv = 'AutoAttack_ckpt'
        # adversarial attack - PGD_Linf
        else:
            self.csv = 'PGD_Attack'

        self.set_tested_model()

        if args.model_path != "":
            # if self.check_is_tested(args.model_path+".pt"):
            #     raise ValueError("This model is already tested")
            self.model_paths = [f'{args.model_path}.pt']
        else:
            exist_model_paths = os.listdir(f'{self.save_path}/')
            self.model_paths = []
            
            for exist_model_path in exist_model_paths:
                if not self.check_is_tested(exist_model_path) and exist_model_path[-2:] == 'pt':
                    self.model_paths.append(exist_model_path)

            # print(self.model_paths)
            # exit()

        self.set_dataset(args)
        self.attack_method()

    def set_tested_model(self):
        self.tested_model_paths = []
        f = open(f'./results/csvs/{self.csv}.csv', 'r', encoding='utf-8')
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

    def attack_method(self): # Atttackckckckckkckckck
        if "AA" in self.test_method:
            self.test_at = AutoAttack(self.model, eps=self.test_eps, args=self.args)
        elif self.test_method == 'bound':
            # self.test_at = PGDAttack(self.model, eps=self.test_eps, alpha=2., iter=20, restart=1, norm='Linf')
            self.test_at = FGSM(self.model, self.test_eps, 0., self.test_eps, initial='none')
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

        self.test_data = CIFAR10(root='./data', train=False, download=False, transform=transform)

        self.test_loader = DataLoader(self.test_data, batch_size=args.batch_size, shuffle=True, num_workers=0)


    def test(self):
        for model_path in self.model_paths:
            if model_path[-2:] != 'pt':
                continue
            self.model.load_state_dict(torch.load(f'{self.save_path}/{model_path}', map_location=self.device))
            self.model.eval()

            if self.test_method == 'bound':
                adv_losses = 0
                lower_bound = 0
                approx_losses = 0
                approx_metrics = 0

                for _, (x, y) in enumerate(tqdm(self.test_loader, desc='bound test')):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    logit = self.model(x)
                    softmax = F.softmax(logit, dim=1)
                    y_onehot = F.one_hot(y, num_classes = softmax.shape[1])
                    loss = F.cross_entropy(logit, y, reduction='none')

                    x_adv = self.test_at.perturb(x, y) # PGD50
                    f_logit = self.model(x_adv)
                    f_norm = torch.norm(f_logit-logit, dim=1)
                    approx_loss = loss + torch.sum((f_logit-logit)*(softmax-y_onehot), dim=1) + self.lipschitz/2.0*torch.pow(f_norm, 2)
                    approx_losses += approx_loss.mean().item()

                    bounded_loss = loss - 1.0/(2*self.lipschitz)*torch.pow(torch.norm(softmax-y_onehot, dim=1),2)
                    lower_bound += bounded_loss.mean().item()
                    
                    approx_metric = torch.abs(torch.norm(f_logit - (logit - 1.0/self.lipschitz*(softmax-y_onehot)), dim=1))
                    approx_metrics += approx_metric.mean().item()/torch.abs(logit).mean().item()

                    adv_losses += F.cross_entropy(f_logit, y).item()

                adv_losses = round(adv_losses/len(self.test_data), 4)
                lower_bound = round(lower_bound/len(self.test_data), 4)
                approx_losses = round(approx_losses/len(self.test_data), 4)
                approx_metrics = approx_metrics/len(self.test_data)

                result = [model_path, 'FGSM', self.test_eps, adv_losses, lower_bound, approx_losses, approx_metrics]
                with open(f'./results/csvs/{self.csv}.csv', 'a', encoding='utf-8', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow(result)

            else:
            
                test_correct_count = 0
                test_adv_correct_count = 0
                for _, (x, y) in enumerate(tqdm(self.test_loader, desc='AA test')):
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
                with open(f'./results/csvs/{self.csv}.csv', 'a', encoding='utf-8', newline='') as f:
                    wr = csv.writer(f)
                    wr.writerow(result)

    def inference(self, x, y):
        logit = self.model(x)
        pred = F.softmax(logit, dim=1)
        outputs = torch.argmax(pred, dim=1)
        correct_count = (outputs == y).sum().item()

        return correct_count