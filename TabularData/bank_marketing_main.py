from util import check_data, read_data, split_labels, adjust_dataset_size, print_result
from train_eval import train
from sklearn.model_selection import train_test_split, StratifiedKFold
import category_encoders
import yaml
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from feature_distribution import FeatureType
from args import get_args, complete_cfg_by_args

args = get_args()


def data_preprocess_global(dst, selected_labels, y_name):
    if selected_labels is not None:
        dst = dst[(dst[y_name] == selected_labels[0]) | (dst[y_name] == selected_labels[1])]
    dst.replace('unknown', np.NaN, inplace=True)
    dst.dropna(inplace=True)
    duration = dst["duration"]
    dst.drop(['default'], axis=1, inplace=True)
    dst.drop(['duration'], axis=1, inplace=True)
    dst.drop(['emp.var.rate', 'nr.employed'], axis=1, inplace=True)
    dst["campaign"] = dst["campaign"].apply(lambda x: 8 if x > 8 else x)
    dst["previous"] = dst["previous"].apply(lambda x: 2 if x >= 2 else x)
    dst.drop(['campaign'], axis=1, inplace=True)
    dst.drop(['previous', 'pdays'], axis=1, inplace=True)

    feature_enc = category_encoders.OrdinalEncoder(cols=['contact', 'poutcome' , 'job', 'month', 'marital', 'day_of_week',
        'education', 'housing', 'loan'])
    # feature_enc = category_encoders.OrdinalEncoder(cols=['housing', 'loan'])
    label_enc = category_encoders.OrdinalEncoder(cols=['y'],
                                     mapping=[{'col': 'y','mapping': {selected_labels[0]: 0, selected_labels[1]: 1}}])
    dst = feature_enc.fit_transform(dst)
    dst = label_enc.fit_transform(dst)

    # contact = pd.get_dummies(dst.contact, drop_first=True)
    # poutcome = pd.get_dummies(dst.poutcome, drop_first=True)
    # job = pd.get_dummies(dst.job, drop_first=True)
    # month = pd.get_dummies(dst.month, drop_first=True)
    # marital = pd.get_dummies(dst.marital, drop_first=True)
    # day_of_week = pd.get_dummies(dst.day_of_week, drop_first=True)
    # education = pd.get_dummies(dst.education, drop_first=True)
    # dst = pd.concat([dst, contact, poutcome, job, month, marital, day_of_week, education], axis=1)
    # dst.drop(['contact', 'poutcome', 'job', 'month', 'marital', 'day_of_week', 'education'], axis=1, inplace=True)

    check_data(dst)

    return dst


def data_preprocess_local(dst, selected_labels, var_categorical:list, var_numerical:list):
    import warnings
    warnings.filterwarnings('ignore')
    scaler = MinMaxScaler()
    dst[var_numerical] = scaler.fit_transform(dst[var_numerical])
    return dst


def main():
    cfg_path = 'yaml_config/bank_marketing.yaml'
    # seed_set = [10*i for i in range(10)]
    np.random.seed(0)
    seed_set = np.random.randint(0, 10000, size=100).tolist()
    selected_labels = ['no', 'yes']
    var_categorical = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week",
                       "poutcome"]
    var_numerical = ["age","cons.price.idx","cons.conf.idx", "euribor3m"]

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg = complete_cfg_by_args(cfg, args)
        print(cfg)

    dst = read_data(cfg['file_path'], delimiter=';')

    dst = data_preprocess_global(dst, selected_labels, y_name='y')

    dst = adjust_dataset_size(dst, action_type=1, y_name='y', sample_rate=0.1)

    print(len(dst))
    print(dst.shape)
    return

    # dst = adjust_dataset_size(dst, action_type=2, y_name='y', unbalanced_ratio=15)

    dst_x, dst_y = split_labels(dst, y_name='y')

    x_train, x_test, y_train, y_test = train_test_split(dst_x, dst_y, test_size=cfg['test_ratio'], random_state=100,
                                                        stratify=dst_y)

    x_train = data_preprocess_local(x_train, selected_labels, var_categorical, var_numerical)
    x_test = data_preprocess_local(x_test, selected_labels, var_categorical, var_numerical)

    feature_cols = FeatureType(
        categorical_f=['contact', 'poutcome' , 'job', 'month', 'marital', 'day_of_week',
          'housing', 'loan'],
        numerical_f=var_numerical,
        oridinal_f=['education']
    )
    train(cfg, seed_set, (x_train, y_train), (x_test, y_test), feature_cols, label_list=[0, 1])

    # print(x_train.shape, y_train.shape)
    #
    # from analysis import FeatureDistribution
    # fd = FeatureDistribution(x_train, y_train, labels=[0, 1],
    #                      categorical_col=['contact', 'poutcome' , 'job', 'month', 'marital', 'day_of_week',
    #     'education', 'housing', 'loan'],
    #                      numerical_col=var_numerical)
    #
    # fd.run()
    # analysis
    # from analysis import tsne_vis
    # tsne_vis(x_train.values, y_train)
    # tsne_vis(x_train.values, y_train)
    # return 0
    # check correctness
    # print(x_train.head())
    # print(x_train.head().index)
    # print(dst_x.iloc[x_train.head().index.tolist()])

    # perf_dict = train(cfg, seed_set, x_train, x_test, y_train, y_test)
    # print_result(perf_dict)


if __name__ == '__main__':
    main()

