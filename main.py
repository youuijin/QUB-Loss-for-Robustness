import  torch, argparse

from pyprnt import prnt

from utils import train_utils
from modules.Manager import Manager
from modules.Trainer import Trainer
from modules.Tester import Tester

def main(args):
    ## set seed ##
    train_utils.set_seed()

    ## gpu setting ##
    device = torch.device(f'cuda:{args.device_num}') if args.device_num>=0 else 'cpu'

    ## model setting ##
    model = train_utils.set_model(args.model, args.n_way, args.imgc, args.pretrained)
    model = model.to(device)

    if args.mode == 'train':
        prnt(vars(args))
        ## logger ##
        manager = Manager(args)

        ## Train ##
        trainer = Trainer(args, model, device, manager)

        last_val, last_val_adv, train_time, attack_time = trainer.train()
        test_acc, test_adv_acc = trainer.test()

        # ## logging result in csv ##
        result = [manager.log_name, test_acc, test_adv_acc, last_val, last_val_adv, train_time, attack_time]
        manager.record_csv('result', result)
        
    else:
        # test using auto attack
        tester = Tester(args, model, device)
        tester.test()
        

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--mode', type=str, help='train or test', default='train')
    
    # Common arguments
    ## Dataset options
    argparser.add_argument('--dataset', type=str, help='dataset', default='cifar10')
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=32)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--n_way', type=int, help='n way', default=10)
    argparser.add_argument('--normalize', type=str, default='none')

    ## Model options
    argparser.add_argument('--model', type=str, help='type of model to use', default="resnet18")
    argparser.add_argument('--pretrained', action='store_true', help='loading pretrained model', default=False)
    argparser.add_argument('--save_ckpt', action='store_true', help='saving model in every validation phase', default=False)
    
    ## GPU options
    argparser.add_argument('--device_num', type=int, help='which gpu to use', default=0)

    # Train mode arguments
    ## Training options
    argparser.add_argument('--epoch', type=int, help='epoch number', default=150)
    argparser.add_argument('--batch_size', type=int, help='batch size in epoch', default=128)
    argparser.add_argument('--lr', type=float, help='learning rate', default=0.1)
    argparser.add_argument('--sche', type=str, default="multistep")
    argparser.add_argument('--loss', type=str, default='CE')

    ## adversarial attack options
    argparser.add_argument('--train_attack', type=str, help='attack for adversarial training', default="")
    argparser.add_argument('--train_eps', type=float, help='training attack bound', default=8.0)
    argparser.add_argument('--val_attack', type=str, default="PGD_Linf")
    argparser.add_argument('--val_eps', type=float, help='validation attack bound', default=8.0)

    ## TRADES options
    argparser.add_argument('--beta', type=float, default=6.)   

    ## PGD Attack options
    argparser.add_argument('--iter', type=int, default=10)
    argparser.add_argument('--norm', type=str, default='Linf')
    argparser.add_argument('--restart', type=int, default=1)

    ## Single Step Attack options
    argparser.add_argument('--a1', type=float, default=4.)
    argparser.add_argument('--a2', type=float, default=8.)    
    argparser.add_argument('--lr_att', type=float, default=0.001)

    ## QAUB Attack options
    argparser.add_argument('--lipschitz', type=float, default=0.5)
    argparser.add_argument('--step', type=int, default=0)

    # Test mode arguments
    argparser.add_argument('--test_method', type=str, help='AA: auto attack, PGD: PGD_Linf attack', default="AA")
    argparser.add_argument('--model_path', type=str, help='test model path', default="")
    argparser.add_argument('--test_eps', type=float, help='test attack bound', default=8.0)

    ## auto attack options
    argparser.add_argument('--auto_version', type=str, help='auto attack version, standard, plus, rand, custom', default='standard')
    argparser.add_argument('--auto_norm', type=str, help='norm for auto attack', default='Linf')
    argparser.add_argument('--auto_custom', type=str, help='if custom version, select auto attack index to test model', default="1000")


    args = argparser.parse_args()

    main(args)