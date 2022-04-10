from util import check_data, read_data, split_labels, adjust_dataset_size, read_config, generate_seed_set
from train_eval import train
from sklearn.model_selection import train_test_split
import category_encoders
from feature_distribution import FeatureType
from args import get_args

args = get_args()


def data_preprocess_global(dst, selected_labels, y_name):
    if selected_labels is not None:
        dst = dst[(dst[y_name] == selected_labels[0]) | (dst[y_name] == selected_labels[1])]
    dst.dropna(inplace=True)

    feature_enc = category_encoders.OrdinalEncoder(cols=list(set(dst.columns.tolist())-set([y_name])))
    label_enc = category_encoders.OrdinalEncoder(cols=[y_name],
                                     mapping=[{'col': y_name,'mapping': {selected_labels[0]: 0, selected_labels[1]: 1}}])
    dst = feature_enc.fit_transform(dst)
    dst = label_enc.fit_transform(dst)

    check_data(dst)

    return dst


def data_preprocess_local(dst, selected_labels, var_categorical:list, var_numerical:list):
    return dst


def main():
    cfg = read_config(cfg_path='yaml_config/mushroom.yaml', args=args)
    seed_set = generate_seed_set()
    selected_labels = ['e', 'p']
    var_categorical = []
    var_numerical = []

    dst = read_data(cfg['file_path'])

    dst = data_preprocess_global(dst, selected_labels, y_name='class')

    dst = adjust_dataset_size(dst, action_type=1, y_name='class', sample_rate=0.1)

    dst_x, dst_y = split_labels(dst, y_name='class')

    x_train, x_test, y_train, y_test = train_test_split(dst_x, dst_y, test_size=cfg['test_ratio'], random_state=42,
                                                        stratify=dst_y)

    x_train = data_preprocess_local(x_train, selected_labels, var_categorical, var_numerical)
    x_test = data_preprocess_local(x_test, selected_labels, var_categorical, var_numerical)

    print(x_train.shape, y_train.shape)

    feature_cols = FeatureType(
        categorical_f=list(x_train.columns),
        numerical_f=[],
        oridinal_f=[]
    )
    train(cfg, seed_set, (x_train, y_train), (x_test, y_test), feature_cols, label_list=[0, 1])


if __name__ == '__main__':
    main()

