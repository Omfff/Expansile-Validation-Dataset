import numpy as np
import yaml
import os


def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path.decode('utf-8'))
        print(path+' create success!')
        return True
    else:
        print(path+' existed!')
        return False


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

    def get_cifar10_dataset_path(self):
        return self.get_dataset_path()

    def get_cifar10_index_path(self):
        return self.get_data_pool_path()+'cifar10_indexes/'

    def get_cifar10_data_pool_path(self):
        return self.get_data_pool_path() + 'cifar10_pool/'

    def get_cifar10_fe_path(self):
        self.get_fe_path()


def generate_seed_set():
    """ Generate 100 random seeds

    :return: list contains 100 seeds
    """
    np.random.seed(0)
    seed_set = np.random.randint(0, 10000, size=100).tolist()
    return seed_set


if __name__ == '__main__':
    def create_folder_in_config():
        pc = PathConfig()
        mkdir(pc.get_cifar10_dataset_path())
        mkdir(pc.get_cifar10_data_pool_path())
        mkdir(pc.get_cifar10_index_path())
        mkdir(pc.get_cifar10_fe_path())
    create_folder_in_config()

