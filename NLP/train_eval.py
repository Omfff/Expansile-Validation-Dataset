from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from datasets import load_metric
import numpy as np
from torch.utils.data import DataLoader
import torch
from utils import split_train_val
from feature_distribution import FeatureDistribution, FeatureExtractorType
from data_extender import DataExtender
from dataset_pool import post_process, get_dataset


def setup_seed(seed):
    import random
    print("Random Seed: ", seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


def eval(model, eval_dataloader, device):
    metric_acc = load_metric("accuracy.py")
    metric_f1 = load_metric("f1.py")
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric_acc.add_batch(predictions=predictions, references=batch["labels"])
            metric_f1.add_batch(predictions=predictions, references=batch["labels"])

    acc = metric_acc.compute()
    f1 = metric_f1.compute(average="macro")
    print("acc", acc)
    print("macro", f1)
    return acc['accuracy'], f1['f1']


def train(model, train_dataloader, eval_dataloader, test_loader, optimizer, lr_scheduler, num_epochs, device):
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)))
    best_acc, best_val_f1, test_f1 = 0, 0, 0
    final_bias_f1 = 0
    for epoch in range(num_epochs):
        if epoch % 2 == 0:
            if eval_dataloader is not None:
                acc, f1 = eval(model, eval_dataloader, device)
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_acc = acc
                    test_acc, test_f1 = eval(model, test_loader, device)
                    print('test score', test_acc, test_f1)
                    final_bias_f1 = abs(best_val_f1-test_f1)
                    print('bias ', final_bias_f1)
            else:
                acc, f1 = eval(model, train_dataloader, device)
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_acc = acc

        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    print("best_acc", best_acc)
    print("best_val_f1", best_val_f1)
    print("test_f1", test_f1)
    print(final_bias_f1)
    return test_f1, best_val_f1, final_bias_f1, model


def train_efficient(model, train_dataloader, eval_dataloader, test_loader, optimizer, lr_scheduler, num_epochs, device):
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)))
    best_acc, best_val_f1, test_f1 = 0, 0, 0
    final_bias_f1 = 0
    best_state_dict = None
    for epoch in range(num_epochs):
        if epoch % 2 == 0:
            train_acc, train_f1 = eval(model, train_dataloader, device)
            print("training acc %f and f1 %f"%(train_acc, train_f1))
            if eval_dataloader is not None:
                acc, f1 = eval(model, eval_dataloader, device)
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_acc = acc
                    best_state_dict = model.state_dict()
            else:
                acc, f1 = eval(model, train_dataloader, device)
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    best_acc = acc

        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    model.load_state_dict(best_state_dict)
    test_acc, test_f1 = eval(model, test_loader, device)
    print('test score', test_acc, test_f1)
    final_bias_f1 = abs(best_val_f1 - test_f1)
    print('bias ', final_bias_f1)
    print("best_acc", best_acc)
    print("best_val_f1", best_val_f1)
    print("test_f1", test_f1)
    print(final_bias_f1)
    return test_f1, best_val_f1, final_bias_f1, model


def one_round_training(args, train_args,
                       seed, whole_train_dst, test_dst, num_classes, fe_weight_path=None, data_extender_args=None,
                       data_pool_path = None, device= "cuda:0"):
    model_name = train_args["model_name"]
    num_k = args.k

    test_f1_list = []
    val_f1_list = []
    f1_bias_list = []

    fe_type = FeatureExtractorType.FineTune if args.fe_type=='fine-tune' else FeatureExtractorType.PreTrain
    feature_distributor = FeatureDistribution(labels=[i for i in range(num_classes)], device=device,
                                              weight_path=fe_weight_path,
                                              dis_type=args.feature_dis_type,
                                              feature_extractor_type=fe_type)

    if args.syn_val:
        syn_val_set = get_dataset('augmented_imdb_valset_byorder', model_name, seed)
    else:
        train_val_index_list = split_train_val([i for i in range(len(whole_train_dst))], whole_train_dst['labels'],
                                               seed=seed, k=num_k, val_ratio=0.2)

    # np.random.seed(seed)
    # data_extender_seeds = np.random.randint(0, 10000, size=num_k).tolist()
    # print(data_extender_seeds)
    for i in range(num_k):
        if args.syn_val:
            train_dst = whole_train_dst
            val_dst = syn_val_set
        else:
            train_index, val_index = train_val_index_list[i]
            train_dst = whole_train_dst.select(train_index)
            val_dst = whole_train_dst.select(val_index)

        if 'aug' in args.val_method:
            train_val_feature_distribution_diff = feature_distributor.cal_distribution_diff_for_two_set(
                val_dst.remove_columns(['labels']), val_dst['labels'], whole_train_dst.remove_columns(['labels']),
                whole_train_dst['labels'])
            print('train_val_feature_distribution_diff', train_val_feature_distribution_diff)
            data_extender = DataExtender(whole_train_dst, train_dst, val_dst, fd=feature_distributor, random_seed=seed,
                                         **data_extender_args)
            data_extender.generate_data_to_pool(data_pool_path, post_process, model_name)
            train_dst, val_dst = data_extender.run(ignore_feature_distance=args.ignore_fdd)

        setup_seed(42)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        model.to(device)
        train_dst.set_format("torch")
        val_dst.set_format("torch")
        test_dst.set_format("torch")
        train_dataloader = DataLoader(train_dst, shuffle=True, batch_size=train_args["batch_size_train"])
        eval_dataloader = DataLoader(val_dst, batch_size=train_args["batch_size_eval"])
        test_dataloader = DataLoader(test_dst, batch_size=train_args["batch_size_eval"])

        optimizer = AdamW(model.parameters(), lr=train_args["lr"], weight_decay=train_args["weight_decay"])
        num_epochs = train_args["num_epochs"]
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_epochs * len(train_dataloader)
        )
        test_f1, val_f1, f1_bias, _ = train_efficient(model, train_dataloader, eval_dataloader, test_dataloader, optimizer,
                                            lr_scheduler, num_epochs, device)
        if abs(abs(test_f1-val_f1) - f1_bias) > 0.0000001:
            raise Exception("test f1 - val f1 not match f1_bias")
        test_f1_list.append(test_f1)
        val_f1_list.append(val_f1)
        f1_bias_list.append(f1_bias)

        del model
        torch.cuda.empty_cache()
    return np.mean(test_f1_list), np.mean(val_f1_list), np.mean(f1_bias_list)


def Kfold_cross_validation(args, one_round_training_func, one_round_training_args):
    np.random.seed(0)  # 0
    seed_set = np.random.randint(0, 10000, size=10).tolist()
    val_performance_list = []
    test_performance_list = []
    performance_bias_list = []
    for i, s in enumerate(seed_set):
        print("=" * 20 + str(i) + "=" * 20)
        test_perf, val_perf, perf_bias = one_round_training_func(seed=s, **one_round_training_args)
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


def JKfold_cross_validation(args, one_round_training_func, one_round_training_args):
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
            test_perf, val_perf, perf_bias = one_round_training_func(seed=seeds_for_kfold[j], **one_round_training_args)
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
