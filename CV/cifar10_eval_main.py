import os
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import models
from args import get_cifar10_args
from data_loader import get_dataloader, split_train_val
from imbalanced_dataset import get_dataset, DatasetWrapper, update_transform
from train_eval import train, setup_seed
from feature_distribution import FeatureDistribution, FeatureExtractorType, FeatureExtractor
from data_extender import DataExtender
from feature_extractor import load_feature_extractor
import yaml
from utils import generate_seed_set


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(models.__dict__[name]))

print(model_names)

args = get_cifar10_args(model_names)
print(args)
# setup_seed(seed=42)

with open(args.config_path, 'r') as f:
    cfg = yaml.safe_load(f)
    print(cfg)


def one_round_training(seed):
    device = "cuda:0"
    num_k = args.k
    num_classes = 10

    test_perf_list = []
    val_perf_list = []
    perf_bias_list = []

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    whole_train_dst, test_dst = get_dataset('im_cifar10', 'train'), get_dataset('im_cifar10', 'test')

    if args.val_method == 'coreset_whole' or args.val_method == 'coreset_part_holdout':
        init_val_dst = get_dataset('coreset_cifar10',index=seed)
    elif args.val_method == 'random_coreset':
        init_val_dst = get_dataset('byorder_valset_cifar10', index=seed)

    update_transform(test_dst, t_type='test')
    update_transform(whole_train_dst, t_type='test')
    train_val_index_list = split_train_val(whole_train_dst.indexset,
                                           whole_train_dst.get_label_list(), seed=seed, k=num_k, val_ratio=0.2)

    if 'aug' in args.val_method:
        fe_type = FeatureExtractorType.FineTune if args.fe_type == 'fine-tune' else FeatureExtractorType.PreTrain
        feature_distributor = FeatureDistribution(labels=[i for i in range(num_classes)],
                                                  feature_extractor=FeatureExtractor(fe_type, load_feature_extractor(
                                                      'feature_extractor/model.th', device), device),
                                                  dis_type=args.feature_dis_type)
    for i in range(num_k):
        print("="*20+"curr k "+str(i)+'='*20)
        # === prepare data begin ===
        if args.val_method == 'coreset_whole' or args.val_method == 'random_coreset':
            train_dst = whole_train_dst
            val_dst = init_val_dst
        else:
            train_index, val_index = train_val_index_list[i]
            train_dst = DatasetWrapper(whole_train_dst.get_dataset_by_indexes(train_index, has_transform=None))
            if args.val_method == 'coreset_part_holdout':
                val_dst = init_val_dst
            else:
                val_dst = DatasetWrapper(whole_train_dst.get_dataset_by_indexes(val_index, has_transform=None))

        update_transform(val_dst, t_type='test')

        if 'aug' in args.val_method:
            train_val_feature_distribution_diff = feature_distributor.\
                cal_distribution_diff_for_two_set(val_dst, None, whole_train_dst, None)
            print('train_val_feature_distribution_diff', train_val_feature_distribution_diff)
            data_extender = DataExtender(whole_train_dst, train_dst, val_dst, fd=feature_distributor,
                                         iter_num=cfg["iter_num"],
                                         add_ratio_per_iter=cfg["add_ratio_per_iter"],
                                         diff_threshold_ratio=(cfg["diff_threshold_ratio"] * num_classes),
                                         early_stop_threshold=cfg["early_stop_threshold"],
                                         try_num_limits=cfg["try_num_limits"],
                                         add_num_decay_rate=cfg["add_num_decay_rate"],
                                         add_num_decay_method=cfg["add_num_decay_method"],
                                         add_num_decay_stage=cfg["add_num_decay_stage"], random_seed=seed)
            data_extender.generate_data_to_pool('aug_cifar10')
            train_dst, val_dst = data_extender.run(ignore_feature_distance=args.ignore_fdd)

        print('train set size %d, val set size %d'%(len(train_dst), len(val_dst)))
        setup_seed(seed=42)

        update_transform(train_dst, t_type='train')
        train_loader, val_loader, test_loader = get_dataloader(args, (train_dst, val_dst), test_dst)
        # === prepare data end ===

        # === training module set up ===
        model = models.__dict__[args.arch]()
        model.to(device)

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
        if args.arch in ['resnet1202', 'resnet110']:
            # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
            # then switch back. In this setup it will correspond for first epoch.
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1

        # === training ===
        test_perf, best_val_perf, val_test_bias = train(train_loader, val_loader, test_loader,
              model, criterion, optimizer, lr_scheduler,
              args.epochs, device, save_dir=args.save_dir)

        # === recording ===
        if abs(abs(test_perf - best_val_perf) - val_test_bias) > 0.0000001:
            raise Exception("test perf - val perf not match val_test_bias")
        test_perf_list.append(test_perf)
        val_perf_list.append(best_val_perf)
        perf_bias_list.append(val_test_bias)

        # === clear ===
        del model
        torch.cuda.empty_cache()

    return np.mean(test_perf_list), np.mean(val_perf_list), np.mean(perf_bias_list)


def Kfold_cross_validation():
    import numpy as np
    seed_set = generate_seed_set()
    val_performance_list = []
    test_performance_list = []
    performance_bias_list = []
    for i, s in enumerate(seed_set):
        print("="*20+str(i)+"="*20)
        test_perf, val_perf, perf_bias = one_round_training(s)
        val_performance_list.append(val_perf)
        test_performance_list.append(test_perf)
        performance_bias_list.append(perf_bias)

    print(args)
    print(val_performance_list)
    print(test_performance_list)
    print(performance_bias_list)
    print("val average performance", np.mean(val_performance_list))
    print("test average performance", np.mean(test_performance_list))
    print("val performance std ", np.std(val_performance_list))
    print("performance bias ", np.mean(performance_bias_list))

    with open(args.save_name, 'wb') as f:
        pickle.dump({'args':args,
                    'val_performance_list':val_performance_list,
                     'test_performance_list':test_performance_list,
                     'performance_bias_list':performance_bias_list}, f)


def JKfold_cross_validation():
    repeat_j = 4 # 2
    import numpy as np
    np.random.seed(0)
    # 5 round to get average
    seed_set = np.random.randint(0, 10000, size=5).tolist()
    seeds_for_kfold_list = []
    for s in seed_set:
        np.random.seed(s)
        seeds_for_kfold_list.append(np.random.randint(0, 10000, size=repeat_j).tolist())

    val_performance_list = []
    test_performance_list = []
    performance_bias_list = []

    for i in range(len(seeds_for_kfold_list)):
        jk_val_performance_list = []
        jk_test_performance_list = []
        jk_performance_bias_list = []
        print("=" * 20 + str(i) + "=" * 20)
        seeds_for_kfold = seeds_for_kfold_list[i]
        for j in range(repeat_j):
            test_perf, val_perf, perf_bias = one_round_training(seeds_for_kfold[j])
            jk_val_performance_list.append(val_perf)
            jk_test_performance_list.append(test_perf)
            jk_performance_bias_list.append(perf_bias)

        val_performance_list.append(np.mean(jk_val_performance_list))
        test_performance_list.append(np.mean(jk_test_performance_list))
        performance_bias_list.append(np.mean(jk_performance_bias_list))

    print(args)
    print(val_performance_list)
    print(test_performance_list)
    print(performance_bias_list)
    print("val average performance", np.mean(val_performance_list))
    print("test average performance", np.mean(test_performance_list))
    print("val performance std ", np.std(val_performance_list))
    print("performance bias ", np.mean(performance_bias_list))

    with open(args.save_name, 'wb') as f:
        pickle.dump({'args': args,
                     'val_performance_list': val_performance_list,
                     'test_performance_list': test_performance_list,
                     'performance_bias_list': performance_bias_list}, f)

def main():
    if args.J == 1:
        Kfold_cross_validation()
    elif args.J > 1:
        JKfold_cross_validation()
    else:
        raise Exception("J value error %d"%args.J)


if __name__ == '__main__':
    main()
