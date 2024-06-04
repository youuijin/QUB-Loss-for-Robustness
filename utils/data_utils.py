import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

from torch.utils.data import DataLoader, random_split

def set_dataloader(args):
    ### normalize setting ###
    if args.normalize == "imagenet":
        norm_mean, norm_std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    elif args.normalize == "twice":
        norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    else: 
        norm_mean, norm_std = (0, 0, 0), (1, 1, 1)

    ### dataset ###
    transform = transforms.Compose([transforms.RandomCrop(args.imgsz, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)])
    transform_test = transforms.Compose([transforms.CenterCrop((args.imgsz, args.imgsz)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])

    if args.dataset == 'cifar10':
        train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_data = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif args.dataset == 'cifar100':
        train_data = CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_data = CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    elif args.dataset == 'svhn':
        train_data = SVHN(root='./data', split='train', download=True, transform=transform)
        test_data = SVHN(root='./data', split='test', download=True, transform=transform)

    train_size = int(len(train_data) * 0.8) # 80% training data
    valid_size = len(train_data) - train_size # 20% validation data
    train_data, valid_data = random_split(train_data, [train_size, valid_size])
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

    return train_loader, val_loader, test_loader