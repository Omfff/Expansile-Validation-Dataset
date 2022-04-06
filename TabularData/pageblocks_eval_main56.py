import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import category_encoders
from util import adjust_dataset_size
from train_eval import train
import yaml
from feature_distribution import FeatureType
from args import get_args, complete_cfg_by_args

args = get_args()


def check_data(data):
    print(data.shape)
    print(data.head())
    print(data.isnull().sum())
    print(data.info())
    for col in data.columns.values:
        print(data[col].value_counts())
    print(data.describe())


def data_preprocessing(data):
    max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    z_score_scaler = lambda x: (x - np.mean(x)) / np.std(x)
    scaler = z_score_scaler
    data[['height']] = data[['height']].apply(scaler)
    data[['lenght']] = data[['lenght']].apply(scaler)
    data[['area']] = data[['area']].apply(scaler)
    data[['blackpix']] = data[['blackpix']].apply(scaler)
    data[['blackand']] = data[['blackand']].apply(scaler)
    data[['wb_trans']] = data[['wb_trans']].apply(scaler)
    return data


def main():
    cfg_path = 'yaml_config/pageblocks42.yaml'
    np.random.seed(0) #0
    seed_set = np.random.randint(0, 10000, size=100).tolist()
    # seed_set = [10 * i for i in range(10)]
    selected_labels = [1, 4] #4

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = complete_cfg_by_args(cfg, args)
        print(cfg)

    dst = pd.read_csv(cfg['file_path'])
    check_data(dst)
    dst = dst[(dst['class'] == selected_labels[0]) | (dst['class'] == selected_labels[1])]
    print(len(dst))
    print(dst.shape)
    return
    # dst = adjust_dataset_size(dst, action_type=3, y_name='class', unbalanced_ratio=50)
    dst_x = dst.drop(['class'], axis=1)
    dst_y = dst['class']

    x_train, x_test, y_train, y_test = train_test_split(dst_x, dst_y, test_size=cfg['test_ratio'], random_state=42,
                                                        stratify=dst_y)
    print(x_train.shape, y_train.shape)

    x_train = data_preprocessing(x_train)
    x_test = data_preprocessing(x_test)
    # print(x_train.head())

    target_enc = category_encoders.OrdinalEncoder(cols=['acceptability'],
                                                  mapping=[
                                                      {'col': 'class',
                                                       'mapping': {selected_labels[0]: 0, selected_labels[1]: 1}}])
    y_train = target_enc.fit_transform(y_train)
    y_test = target_enc.transform(y_test)

    print(y_train.value_counts())

    feature_cols = FeatureType(
        categorical_f=[],
        numerical_f=['height', 'lenght', 'area', 'eccen', 'p_black', 'p_and', 'mean_tr',
                                                'blackpix', 'blackand', 'wb_trans'],
        oridinal_f=[]
    )
    train(cfg, seed_set, (x_train, y_train['class']), (x_test, y_test), feature_cols, label_list = [0, 1])

    # print(x_train.columns)
    # from analysis import FeatureDistribution
    # fd = FeatureDistribution(x_train, y_train['class'], labels=[0, 1],
    #                          categorical_col=[],
    #                          numerical_col=['height', 'lenght', 'area', 'eccen', 'p_black', 'p_and', 'mean_tr',
    #    'blackpix', 'blackand', 'wb_trans'])
    #
    # aug_set = AugmentSet((x_train, y_train['class']), fd=fd, val_ratio=0.2, iter_num=10, add_ratio_per_iter=0.2,
    #                                 diff_threshold_ratio=0.0001, random_seed = 42)
    #
    # aug_set.generate_data_to_pool(method='SMOTE')
    #
    # aug_set.run()


if __name__ == '__main__':
    main()