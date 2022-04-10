import datasets
from nltk.corpus import reuters
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AdamW, get_scheduler
from datasets import load_metric, load_dataset, get_dataset_config_names
import numpy as np
from torch.utils.data import DataLoader
from train_eval import train, setup_seed
import torch
import pandas as pd
from utils import split_train_val
from feature_distribution import FeatureDistribution, FeatureExtractorType
from data_extender import DataExtender
from dataset_pool import get_dataset, post_process
from args import get_args
from utils import PathConfig, generate_seed_set


FE_SAVE_PATH = PathConfig().get_reuters_fe_path()
DATA_POOL_PATH = PathConfig().get_reuters_data_pool_path()

args = get_args()
print(args)


def compute_metrics(eval_pred):
    metric = load_metric("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="macro")


def one_round_training(seed):
    device = "cuda:0"
    model_name = "distilbert-base-uncased"
    dataset_name = 'reuters21578'
    num_classes = 2
    num_k = args.k

    test_f1_list = []
    val_f1_list = []
    f1_bias_list = []

    fe_type = FeatureExtractorType.FineTune if args.fe_type=='fine-tune' else FeatureExtractorType.PreTrain
    feature_distributor = FeatureDistribution(labels=[i for i in range(num_classes)], device=device,
                                              weight_path=FE_SAVE_PATH,
                                              dis_type=args.feature_dis_type,
                                              feature_extractor_type=fe_type)

    whole_train_dst, test_dst = get_dataset('wheat_corn_reuters', model_name)
    train_val_index_list = split_train_val([i for i in range(len(whole_train_dst))], whole_train_dst['labels'], seed=seed, k=num_k, val_ratio=0.2)
    if 'coreset' in args.val_method:
        val_dst = get_dataset('augmented_wheat_corn_reuters_valset', model_name, seed=seed)
    np.random.seed(seed)
    data_extender_seeds = np.random.randint(0, 10000, size=num_k).tolist()
    print(data_extender_seeds)
    for i in range(num_k):
        train_index, val_index = train_val_index_list[i]
        if args.val_method == 'coreset_whole':
            train_dst = whole_train_dst
        else:
            if 'coreset' not in args.val_method:
                val_dst = whole_train_dst.select(val_index)
            train_dst = whole_train_dst.select(train_index)

        if 'aug' in args.val_method:
            train_val_feature_distribution_diff = feature_distributor.cal_distribution_diff_for_two_set(
                val_dst.remove_columns(['labels']), val_dst['labels'], whole_train_dst.remove_columns(['labels']),
                whole_train_dst['labels'])
            print('train_val_feature_distribution_diff', train_val_feature_distribution_diff)
            data_extender = DataExtender(whole_train_dst, train_dst, val_dst, fd=feature_distributor, iter_num=10,
                                         add_ratio_per_iter=0.1, diff_threshold_ratio=(0.01 * num_classes), early_stop_threshold=3,
                                         try_num_limits=150,
                                         add_num_decay_rate=0.5, add_num_decay_method='triggered', add_num_decay_stage=None,
                                         random_seed=data_extender_seeds[i])
            data_extender.generate_data_to_pool(DATA_POOL_PATH+'pool.csv', post_process, model_name)
            train_dst, val_dst = data_extender.run(ignore_feature_distance=args.ignore_fdd)

        setup_seed(42)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels= num_classes)
        print("train dst size %d val dst size %d" % (len(train_dst), len(val_dst)))
        model.to(device)
        train_dst.set_format("torch")
        val_dst.set_format("torch")
        test_dst.set_format("torch")
        train_dataloader = DataLoader(train_dst, shuffle=True, batch_size=8)
        eval_dataloader = DataLoader(val_dst, batch_size=64)
        test_dataloader = DataLoader(test_dst, batch_size=64)

        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        num_epochs = 15
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_epochs * len(train_dataloader)
        )
        test_f1, val_f1, f1_bias, _ = train(model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler, num_epochs, device)
        if abs(abs(test_f1-val_f1) - f1_bias) > 0.0000001:
            raise Exception("test f1 - val f1 not match f1_bias")
        test_f1_list.append(test_f1)
        val_f1_list.append(val_f1)
        f1_bias_list.append(f1_bias)

        del model
        torch.cuda.empty_cache()

    return np.mean(test_f1_list), np.mean(val_f1_list), np.mean(f1_bias_list)


def Kfold_cross_validation():
    seed_set = generate_seed_set()
    val_performance_list = []
    test_performance_list = []
    performance_bias_list = []
    for i, s in enumerate(seed_set):
        print("=" * 20 + str(i) + "=" * 20)
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

    import pickle
    with open(args.save_name, 'wb') as f:
        pickle.dump({'args':args,
                     'val_performance_list':val_performance_list,
                     'test_performance_list':test_performance_list,
                     'performance_bias_list':performance_bias_list}, f)


def JKfold_cross_validation():
    repeat_j = args.J # 2
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
    print(seeds_for_kfold_list)
    print(val_performance_list)
    print(test_performance_list)
    print(performance_bias_list)
    print("val average performance", np.mean(val_performance_list))
    print("test average performance", np.mean(test_performance_list))
    print("val performance std ", np.std(val_performance_list))
    print("performance bias ", np.mean(performance_bias_list))

    import pickle
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




