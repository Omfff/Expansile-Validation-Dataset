import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import yaml

def split_train_val(indexes, y, seed, k=1, val_ratio=0.2):
    labels = y
    train_val_set_list = []
    if k == 1:
        train_indexes, val_indexes, _, _ = train_test_split(indexes, labels, test_size=val_ratio,
                                                            stratify=labels, random_state=seed)
        train_val_set_list.append((train_indexes, val_indexes))
    else:
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        for train_ind, val_ind in kfold.split(indexes, labels):
            train_val_set_list.append((train_ind, val_ind))
    return train_val_set_list


class PathConfig(object):
    def __init__(self):
        cfg_path = "path_config.yaml"
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
            print(cfg)
        self.cfg = cfg

    def get_dataset_path(self):
        return self.cfg['dataset_path']

    def get_data_pool_path(self):
        return self.cfg['data_pool_path']

    def get_fe_path(self):
        return self.cfg['feature_extractor_save_path']

    def get_gs_path(self):
        return self.cfg['grid_search_save_path']


def generate_seed_set():
    """ Generate 100 random seeds

    :return: list contains 100 seeds
    """
    np.random.seed(0)
    seed_set = np.random.randint(0, 10000, size=100).tolist()
    return seed_set

if __name__ == '__main__':
    PC = PathConfig()
    print(PC.get_dataset_path())
    print(PC.get_data_pool_path())
