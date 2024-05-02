from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import csv, torch, os

class Manager:
    def __init__(self, args, log_dir):
        cur = datetime.now().strftime('%m-%d_%H-%M')

        if args.train_attack!="":
            if args.train_attack == 'QAUB':
                log_name = f"{args.model}/{args.train_attack}/step{args.step}/eps{args.train_eps}(lips{args.lipschitz})/lr{args.lr}_{args.sche}/{cur}"
            elif args.train_attack != 'PGD_Linf':
                log_name = f"{args.model}/{args.train_attack}/eps{args.train_eps}({args.a1}_{args.a2})/lr{args.lr}_{args.sche}/{cur}"
            else:
                log_name = f"{args.model}/{args.train_attack}/eps{args.train_eps}/lr{args.lr}_{args.sche}/{cur}"
        else:
            log_name = f"{args.model}/no_attack/lr{args.lr}_{args.sche}/{cur}"

        writer = SummaryWriter(f"./{log_dir}/{log_name}")

        self.log_name = log_name
        self.save_dir = './results'
        self.writer = writer

    def record(self, writer_name, name, value, epoch):
        if writer_name == "writer":
            writer = self.writer

        writer.add_scalar(name, value, epoch)

    def record_csv(self, file_name, rows):
        with open(f'{self.save_dir}/csvs/{file_name}.csv', 'a', encoding='utf-8', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(rows)
    
    def save_model(self, model, ckpt=-1):
        if ckpt >= 0:
            # during valudation phase
            os.makedirs(f'{self.save_dir}/ckpt/{self.log_name.replace("/", "_")}', exist_ok=True)
            save_path=f'{self.save_dir}/ckpt/{self.log_name.replace("/", "_")}/epoch_{ckpt}.pt'
        else:
            # after whole training phase
            save_path = f'{self.save_dir}/saved_model/{self.log_name.replace("/", "_")}.pt'
        torch.save(model.state_dict(), save_path)