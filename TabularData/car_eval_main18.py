import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
import category_encoders
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from util import  adjust_dataset_size
import yaml
from feature_distribution import FeatureType
from args import get_args, complete_cfg_by_args

args = get_args()

def read_data(file_path):
    data = pd.read_csv(file_path)
    return data


def check_data(data):
    print(data.shape)
    print(data.head())
    print(data.isnull().sum())
    print(data.info())
    for col in data.columns.values:
        print(data[col].value_counts())
    print(data.describe())


def data_preprocessing(dst_x, dst_y):
    pass


def train():
    pass


THRESHOLD = 0.5
def xgb_f1(preds, dst):
    y = dst.get_label()
    return 'f1-err', 1- f1_score(y, preds > THRESHOLD)


def main():
    cfg_path = 'yaml_config/car_eval18.yaml'
    np.random.seed(0)
    seed_set = np.random.randint(0, 10000, size=100).tolist()
    selected_labels = ['unacc', 'good']
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = complete_cfg_by_args(cfg, args)
        print(cfg)

    dst = read_data(cfg['file_path'])
    dst = dst[(dst['acceptability'] == selected_labels[0])| (dst['acceptability'] == selected_labels[1])]
    check_data(dst)
    print(len(dst))
    print(dst.shape)
    return


    # dst = adjust_dataset_size(dst, action_type=3, y_name='acceptability', unbalanced_ratio=1)
    dst_x = dst.drop(['acceptability'], axis=1)
    dst_y = dst['acceptability']

    x_train, x_test, y_train, y_test = train_test_split(dst_x, dst_y, test_size=cfg['test_ratio'], random_state=42,
                                                        stratify=dst_y)
    print(x_train.shape, y_train.shape)

    feature_enc = category_encoders.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],
                                       mapping=[
                                           {'col': 'buying', 'mapping': {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4}},
                                           {'col': 'maint', 'mapping': {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4}},
                                           {'col': 'doors', 'mapping': {'2': 1, '3': 2, '4': 3, '5more': 4}},
                                           {'col': 'persons', 'mapping': {'2': 1, '4': 2, 'more': 3}},
                                           {'col': 'safety', 'mapping': {'low': 1, 'med': 2, 'high': 3}},
                                           {'col': 'lug_boot', 'mapping': {'small': 1, 'med': 2, 'big': 3}}
                                       ])
    x_train = feature_enc.fit_transform(x_train)
    x_test = feature_enc.transform(x_test)

    target_enc = category_encoders.OrdinalEncoder(cols=['acceptability'],
                                                  mapping=[
                                                      {'col': 'acceptability',
                                                       'mapping': {selected_labels[0]: 0, selected_labels[1]: 1}}])
    y_train = target_enc.fit_transform(y_train)
    y_test = target_enc.transform(y_test)

    print(y_train.value_counts())
    from train_eval import train
    feature_cols = FeatureType(
                    categorical_f=[],
                    numerical_f=[],
                    oridinal_f=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    )
    train(cfg, seed_set, (x_train, y_train['acceptability']), (x_test, y_test), feature_cols, label_list=[0, 1])

    # print(x_train.columns)
    # from analysis import FeatureDistribution
    # fd = FeatureDistribution(x_train, y_train['acceptability'], labels=[0, 1],
    #                          categorical_col=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],
    #                          numerical_col=[])
    #
    # fd.run()



if __name__ == '__main__':
    main()