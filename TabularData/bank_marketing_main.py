from util import check_data, read_data, split_labels, adjust_dataset_size, read_config, generate_seed_set
from train_eval import train
from sklearn.model_selection import train_test_split
import category_encoders
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from feature_distribution import FeatureType
from args import get_args

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
    cfg = read_config(cfg_path='yaml_config/bank_marketing.yaml', args=args)
    seed_set = generate_seed_set()
    selected_labels = ['no', 'yes']
    var_categorical = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week",
                       "poutcome"]
    var_numerical = ["age","cons.price.idx","cons.conf.idx", "euribor3m"]

    dst = read_data(cfg['file_path'], delimiter=';')

    dst = data_preprocess_global(dst, selected_labels, y_name='y')

    dst = adjust_dataset_size(dst, action_type=1, y_name='y', sample_rate=0.1)

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


if __name__ == '__main__':
    main()

