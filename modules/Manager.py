from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import csv, torch, os

class Manager:
    def __init__(self, args):
        cur = datetime.now().strftime('%m-%d_%H-%M')

        if args.train_attack!="":
            if args.train_attack == 'PGD_Linf':
                attack_name = f'{args.train_attack}/eps{args.train_eps}'
            elif 'FGSM' in args.train_attack:
                attack_name = f'{args.train_attack}/eps{args.train_eps}({args.a1}_{args.a2})'
            else:
                raise ValueError('Attack Name for logger')
            
            log_name = f'{args.model}/{args.loss}/{attack_name}'
           
        else:
            log_name = f"{args.model}/no_attack/lr{args.lr}_{args.sche}/{cur}"

        self.save_dir = f'./results/{args.dataset}'
        writer = SummaryWriter(f"{self.save_dir}/logs/{log_name}")
        self.log_name = log_name
        self.writer = writer

    def record(self, writer_name, name, value, epoch):
        if writer_name == "writer":
            writer = self.writer
            writer.add_scalar(name, value, epoch)
        elif writer_name == "approx":
            writer = self.approx_writer
            writer.add_scalar(self.log_name, value, epoch)
        elif writer_name == "adv":
            writer = self.adv_writer
            writer.add_scalar(self.log_name, value, epoch)

        
    def record_csv(self, file_name, rows):
        with open(f'{self.save_dir}/csvs/{file_name}.csv', 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(rows)
    
    def save_model(self, model, ckpt=-1, mode='best'):
        if mode == 'last':
            save_path = f'{self.save_dir}/saved_model/{self.log_name.replace("/", "_")}_last.pt'
        elif ckpt >= 0:
            # during valudation phase
            str_ckpt = '0'*(3-len(str(ckpt))) + str(ckpt)
            os.makedirs(f'{self.save_dir}/ckpt/{self.log_name.replace("/", "_")}', exist_ok=True)
            save_path=f'{self.save_dir}/ckpt/{self.log_name.replace("/", "_")}/epoch_{str_ckpt}.pt'
        else:
            # after whole training phase
            save_path = f'{self.save_dir}/saved_model/{self.log_name.replace("/", "_")}.pt'
        torch.save(model.state_dict(), save_path)