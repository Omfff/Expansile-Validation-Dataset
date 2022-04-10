import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import category_encoders
from util import read_data, split_labels,read_config, generate_seed_set
from feature_distribution import FeatureType
from args import get_args

args = get_args()


def main():
    cfg = read_config(cfg_path='yaml_config/car_eval18.yaml', args=args)
    seed_set = generate_seed_set()

    selected_labels = ['unacc', 'good']

    dst = read_data(cfg['file_path'])

    dst = dst[(dst['acceptability'] == selected_labels[0])| (dst['acceptability'] == selected_labels[1])]

    dst_x, dst_y = split_labels(dst, y_name='acceptability')

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


if __name__ == '__main__':
    main()