from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import csv, torch

class Manager:
    def __init__(self, args, log_dir):
        cur = datetime.now().strftime('%m-%d_%H-%M')

        if args.train_attack!="":
            if args.alpha != 4.0:
                log_name = f"{args.model}/{args.train_attack}/eps{args.train_eps}_alpha{args.alpha}/lr{args.lr}_{args.sche}/{cur}"
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
    
    def save_model(self, model):
        save_path = f'{self.save_dir}/saved_model/{self.log_name.replace("/", "_")}.pt'
        torch.save(model.state_dict(), save_path)