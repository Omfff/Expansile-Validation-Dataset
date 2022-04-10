import numpy as np
from sklearn.model_selection import train_test_split
import category_encoders
from util import split_labels, read_config, read_data, generate_seed_set
from train_eval import train
from feature_distribution import FeatureType
from args import get_args

args = get_args()


def data_preprocessing_local(data):
    # max_min_scaler = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
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
    cfg = read_config(cfg_path='yaml_config/pageblocks56.yaml', args=args)
    seed_set = generate_seed_set()
    selected_labels = [1, 4]

    dst = read_data(cfg['file_path'])

    dst = dst[(dst['class'] == selected_labels[0]) | (dst['class'] == selected_labels[1])]

    dst_x, dst_y = split_labels(dst, y_name='class')

    x_train, x_test, y_train, y_test = train_test_split(dst_x, dst_y, test_size=cfg['test_ratio'], random_state=42,
                                                        stratify=dst_y)
    print(x_train.shape, y_train.shape)

    x_train = data_preprocessing_local(x_train)
    x_test = data_preprocessing_local(x_test)

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


if __name__ == '__main__':
    main()