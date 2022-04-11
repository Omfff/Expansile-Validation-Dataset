import argparse


def get_cifar10_args(model_names):
    parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

    # model setup
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet44',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                             ' (default: resnet32)')

    # training setup
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    # dataloader setup
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')

    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='experiments/', type=str)

    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    # validation setting
    parser.add_argument('--val_method', '-vm', choices=['holdout', 'kfold', 'jkfold', 'aug_holdout', 'aug_kfold',
                                                        'coreset_part_holdout', 'coreset_whole', 'random_coreset'], required=True)
    parser.add_argument('--J', type=int, default=1, required=False)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--fe_type',  type=str, choices=['fine-tune', 'pre-train'], default='fine-tune')
    parser.add_argument('--feature_dis_type', type=str, choices=['NDB'], required=False)
    parser.add_argument('--ignore_fdd', dest='ignore_fdd', action='store_true')
    parser.add_argument('--save_name', type=str, required=False)
    parser.add_argument('--config_path', type=str, required=False, default='./config/cifar10_default.yaml')

    args = parser.parse_args()
    ignore_fdd = '_ignore_fdd' if args.ignore_fdd else ''
    fe_type = '' if args.fe_type is None else args.fe_type
    feature_dis_type = '' if args.feature_dis_type is None else args.feature_dis_type
    args.save_name = args.save_dir + args.val_method + '_' + fe_type + '_' + feature_dis_type + ignore_fdd + '.pkl'
    args.save_dir = args.save_dir + args.val_method
    return args


def get_cifar10_fe_args(model_names):
    parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

    # model setup
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet110',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                             ' (default: resnet32)')

    # training setup
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    # dataloader setup
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')

    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='feature_extractor/', type=str)

    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')

    args = parser.parse_args()
    print(args)
    return args