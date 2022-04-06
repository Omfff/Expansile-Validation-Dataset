import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import models
from args import get_cifar10_args, get_cifar10_fe_args
from data_loader import get_dataloader, split_train_val
from imbalanced_dataset import get_dataset, update_transform
from train_eval import train, setup_seed


def load_feature_extractor(path, device="cuda:0"):
    checkpoint = torch.load(path)
    model = models.resnet110()#torch.nn.DataParallel(
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    return model


def main():
    model_names = sorted(name for name in models.__dict__
                         if name.islower() and not name.startswith("__")
                         and name.startswith("resnet")
                         and callable(models.__dict__[name]))

    print(model_names)

    args = get_cifar10_fe_args(model_names)
    setup_seed(seed=42)
    device = "cuda:0"
    num_classes = 10

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # === prepare data begin ===
    train_dst, test_dst = get_dataset('im_cifar10', 'train'), get_dataset('im_cifar10', 'test')
    update_transform(test_dst, t_type='test')
    update_transform(train_dst, t_type='train')
    train_loader, val_loader, test_loader = get_dataloader(args, (train_dst, test_dst), test_dst)
    # === prepare data end ===

    # === training module set up ===
    model = models.__dict__[args.arch]()
    model.to(device)
    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        print('update lr for resnet')
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * 0.1

    # === training ===
    train(train_loader, val_loader, test_loader,
                                     model, criterion, optimizer, lr_scheduler,
                                     args.epochs, device, save_dir=args.save_dir)


if __name__ == '__main__':
    # main()
    model  = load_feature_extractor('feature_extractor/model.th', device="cuda:0")
    data = torch.ones((1, 3, 32 ,32), dtype=torch.float).to("cuda:0")
    out = model(data)
    print(out)
