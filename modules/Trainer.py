import  torch
import torch.nn.functional as F
from tqdm import tqdm
import time

from utils import train_utils, data_utils, attack_utils
from attack import PGD


class Trainer:
    def __init__(self, args, model, device, manager):
        self.args = args
        self.model = model
        self.device = device
        self.manager = manager

        ### optimizer, logger setting ###
        self.optim, self.scheduler = train_utils.set_optim(self.model, args.sche, args.lr, args.epoch)
        
        ### data loader setting ###
        self.train_loader, self.val_loader, self.test_loader = data_utils.set_dataloader(args)

    def train(self):
        train_time, tot_attack_time = 0, 0
        last_val, last_val_adv = 0, 0
        best_acc = 0
        
        for epoch in tqdm(range(self.args.epoch), desc='epoch', position=0, ascii=" ="):
            train_correct_count = 0
            train_loss = 0
            real_loss = 0
            # approx_metrics = 0

            self.model.train()

            train_time_st = time.time()
            for _, (x, y) in enumerate(tqdm(self.train_loader, desc='train step', position=1, leave=False, ascii=" =")):
                x = x.to(self.device)
                y = y.to(self.device)
                
                attack = attack_utils.set_attack(self.args.train_attack, self.model, self.args.train_eps, self.args)

                # loss, correct_count, attack_time = attack.calc_loss_acc(x, y) ## original
                # loss, correct_count, attack_time, _ = attack.calc_loss_acc(x, y) ## approximated metric
                if self.args.train_attack == 'QAUB':
                    loss, adv_loss, attack_time = attack.calc_loss_acc(x, y)
                    real_loss += adv_loss.item()*x.shape[0]
                else:
                    loss, correct_count, attack_time = attack.calc_loss_acc(x, y)
                    train_correct_count += correct_count
                
                tot_attack_time += attack_time
                
                train_loss += loss.item()*x.shape[0]
                
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            train_time += (time.time() - train_time_st)

            self.manager.record('writer', "train/acc", round(train_correct_count/len(self.train_loader.dataset)*100, 4), epoch)
            self.manager.record('writer', "train/loss", round(train_loss/len(self.train_loader.dataset), 4), epoch)
            if self.args.train_attack == 'QAUB':
                self.manager.record('approx', "", round(train_loss/len(self.train_loader.dataset), 4), epoch)
                self.manager.record('adv', "", round(real_loss/len(self.train_loader.dataset), 4), epoch)

            if epoch%5==0:
                self.model.eval()
                val_correct_count, val_adv_correct_count = 0, 0
                val_loss, val_adv_loss = 0, 0
                for _, (x, y) in enumerate(tqdm(self.val_loader, desc='val step', position=1, leave=False, ascii=" *")):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    loss, correct_count = self.predict(x, y, self.model)

                    val_correct_count += correct_count
                    val_loss += loss.item() * x.shape[0]
                    
                    # val_attack = attack_utils.set_attack(self.args.val_attack, self.model, self.args.val_eps, self.args)
                    val_attack = PGD.PGDAttack(self.model, self.args.val_eps, iter=10, restart=1)
                    adv_loss, adv_correct_count, _ = val_attack.calc_loss_acc(x, y)

                    val_adv_correct_count += adv_correct_count
                    val_adv_loss += adv_loss.item() * x.shape[0]

                    # approx_metrics += approx_metric

                self.manager.record('writer', "val/acc", round(val_correct_count/len(self.val_loader.dataset)*100, 4), epoch)
                self.manager.record('writer', "val/loss", round(val_loss/len(self.val_loader.dataset)*100, 4), epoch)
                self.manager.record('writer', "val/acc_adv", round(val_adv_correct_count/len(self.val_loader.dataset)*100, 4), epoch)
                self.manager.record('writer', "val/loss_adv", round(val_adv_loss/len(self.val_loader.dataset)*100, 4), epoch)

                last_val = round(val_correct_count/len(self.val_loader.dataset)*100, 4)
                last_val_adv = round(val_adv_correct_count/len(self.val_loader.dataset)*100, 4)

                if self.args.save_ckpt:
                    self.manager.save_model(self.model, epoch)
                ## adv acc 기준 최고 모델 저장
                if best_acc < round(val_adv_correct_count/len(self.val_loader.dataset)*100, 4):
                    best_acc = round(val_adv_correct_count/len(self.val_loader.dataset)*100, 4)
                    self.manager.save_model(self.model)

            self.scheduler.step()

        self.manager.save_model(self.model, mode='last')
            
        return last_val, last_val_adv, round(train_time,4), round(tot_attack_time,4)
    
    def test(self):
        # test
        self.model.eval()
        test_correct_count = 0
        test_adv_correct_count = 0
        val_attack = attack_utils.set_attack(self.args.val_attack, self.model, self.args.val_eps, self.args)
        for _, (x, y) in enumerate(tqdm(self.test_loader, desc="test")):
            x = x.to(self.device)
            y = y.to(self.device)
            _, correct_count = self.predict(x, y, self.model)
            test_correct_count += correct_count
            _, adv_correct_count, _ = val_attack.calc_loss_acc(x, y)
            test_adv_correct_count += adv_correct_count

        test_acc = round(test_correct_count/len(self.test_loader.dataset)*100, 4)
        test_adv_acc = round(test_adv_correct_count/len(self.test_loader.dataset)*100, 4)

        self.manager.record('writer', "test/acc", test_adv_acc*100, test_acc*100)

        return test_acc, test_adv_acc

    def predict(self, x, y, model):
        logit = model(x)
        loss = F.cross_entropy(logit, y)

        pred = F.softmax(logit, dim=1)
        outputs = torch.argmax(pred, dim=1)
        correct_count = (outputs == y).sum().item()
        
        return loss, correct_count