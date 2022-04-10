import pandas as pd
from pandas import DataFrame
import numpy as np
import yaml
from args import complete_cfg_by_args


def read_data(file_path, delimiter=None):
    """ Read dataset in csv form
    :param file_path: dataset file path
    :param delimiter:
    :return: Dataframe
    """
    data = pd.read_csv(file_path, delimiter=delimiter)
    return data


def read_config(cfg_path, args):
    """ Load config file

    :param cfg_path: file path
    :param args:
    :return: config
    """
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = complete_cfg_by_args(cfg, args)
        print(cfg)
    return cfg


def check_data(data):
    """  General inspection of the data.
    :param data:
    :return: None
    """
    print(data.shape)
    print(data.head())
    print(data.isnull().sum())
    print(data.info())
    for col in data.columns.values:
        print(data[col].value_counts())
    print(data.describe())


def split_labels(dst, y_name:str):
    """ Split dataset into data(x) and label(y)

    :param dst: dataset
    :param y_name: column name of label
    :return: datas of dst, labels of dst
    """
    check_data(dst)
    dst_x = dst.drop([y_name], axis=1)
    dst_y = dst[y_name]
    dst_x = dst_x.reset_index(drop=True)
    dst_y = dst_y.reset_index(drop=True)
    return dst_x, dst_y


def generate_seed_set():
    """ Generate 100 random seeds

    :return: list contains 100 seeds
    """
    np.random.seed(0)
    seed_set = np.random.randint(0, 10000, size=100).tolist()
    return seed_set


def adjust_dataset_size(dst:DataFrame, action_type, y_name, sample_rate=None, unbalanced_ratio=None):
    """ adjust the number of samples in a dataset

    :param action_type: 1 sample instance in each class by 'sample_rate'
                        2 fix number of instance in negative class, and down sample positive class by unbalanced ratio
                        3 fix number of instance in positive class, and down sample negtive class by unbalanced ratio
                        4 fix number of instance in negative class, and up sample positive class by unbalanced ratio
    :return: Adjusted dataset
    """
    np.random.seed(seed=2)
    def sampling(group, class_dict):
        name = group.name
        n = class_dict[name]
        return group.sample(n=n)

    class_distri = dst[y_name].value_counts()
    class_dict = {}
    for i in range(len(class_distri)):
        name = class_distri.index[i]
        class_dict[name] = class_distri[name]
    keys = list(class_dict.keys())
    if action_type == 1:
        for key in keys:
            class_dict[key] = int(class_dict[key]*sample_rate)
        dst = dst.groupby(y_name).apply(sampling, class_dict)
    elif action_type ==2:
        class_dict[keys[1]] = int(class_dict[keys[0]]/unbalanced_ratio)
        dst = dst.groupby(y_name).apply(sampling, class_dict)
    elif action_type==3:
        class_dict[keys[0]] = int(class_dict[keys[1]]*unbalanced_ratio)
        # print(class_dict)
        dst = dst.groupby(y_name).apply(sampling, class_dict)
    elif action_type==4:
        class_dict[keys[1]] = int(class_dict[keys[0]] / unbalanced_ratio)
    print('='*80)
    dst.index = dst.index.droplevel()
    print(dst[y_name].value_counts())
    # dst = sklearn.utils.shuffle(dst)
    return dst


def print_result(perf_dict):
    result = {'val_f1_score_mean_std':None, 'val_auc_mean_std':None, 'val_test_bias_mean_mean':None,
              'test_f1_score_mean_mean':None, 'test_auc_mean_mean':None}
    print(perf_dict)

    print('metric-mean std')
    result['val_f1_score_mean_std'] = np.asarray(perf_dict['val_f1_score_mean']).std()
    result['val_auc_mean_std'] = np.asarray(perf_dict['val_auc_mean']).std()
    print('val_f1_score std', result['val_f1_score_mean_std'])
    print('val_auc std', result['val_auc_mean_std'])
    # print('val_test_bias_mean', np.asarray(perf_dict['val_test_bias_mean']).std())
    # print(np.asarray(perf_dict['positive_f1_mean']).std())
    print('metric-bias mean')
    result['val_test_bias_mean_mean'] = np.asarray(perf_dict['val_test_bias_mean']).mean()
    print('val_test_bias mean', result['val_test_bias_mean_mean'])

    print('perf-metric mean')
    result['test_f1_score_mean_mean'] = np.asarray(perf_dict['test_f1_score_mean']).mean()
    result['test_auc_mean_mean'] = np.asarray(perf_dict['test_auc_mean']).mean()
    print('test_f1_score mean', result['test_f1_score_mean_mean'])
    print('test_auc mean', result['test_auc_mean_mean'])

    if len(perf_dict['f1_score_group_std']) > 0:
        print(np.asarray(perf_dict['f1_score_group_std']).mean())
        print(np.asarray(perf_dict['auc_group_std']).mean())
    return result


def merge_dict(x,y):
    for k,v in x.items():
        if k in y.keys():
            y[k] += v
        else:
            y[k] = v
    return y


