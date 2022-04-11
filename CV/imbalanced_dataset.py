import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from utils import PathConfig

PC = PathConfig()


normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])


class IMBALANCECIFAR10Generator(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10Generator, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.index_list = self.gen_imbalanced_data(img_num_list)
        print(self.num_per_cls_dict)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        total_indexes = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            # np.random.seed(0)
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            # print(selec_idx)
            total_indexes.append(selec_idx)
        total_indexes = np.concatenate(total_indexes, axis=0)
        return total_indexes

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def save_index_list(self, path):
        np.savetxt(path, self.index_list, fmt="%d")


class BaseDataset(Dataset):
    def __init__(self, transform=None):
        self.data = None
        self.targets = None
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)


class AugmentedDataset(BaseDataset):
    def __init__(self, file_path, transform=None):
        super().__init__(transform)
        self.__read_data_from_path(file_path)

    def __read_data_from_path(self, path):
        folders = os.listdir(path)
        data = []
        targets = []
        for folder in folders:
            files = os.listdir(os.path.join(path, folder))
            for f in files:
                data.append(np.array(Image.open(os.path.join(os.path.join(path,folder), f))))
                targets.append(folder)
        self.data = np.stack(data, axis=0)
        self.targets = np.asarray(targets, dtype=int)


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, indexes_path, train=True, transform=None, target_transform=None, download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.load_imbalanced_data(indexes_path)
        self.cal_img_num_per_cls()

    def cal_img_num_per_cls(self):
        self.num_per_cls_dict = {}
        classes = np.unique(self.targets)
        for label in classes:
            self.num_per_cls_dict[label] = np.where(self.targets == label)[0].shape[0]

    def load_imbalanced_data(self, indexes_path):
        indexes_list = np.loadtxt(indexes_path, dtype=int)
        new_data = self.data[indexes_list]
        self.data = new_data
        new_targets = np.asarray(self.targets, dtype=int)[indexes_list]
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class DatasetWrapper(Dataset):
    def __init__(self, dataset, indexset:list=None, transform:transforms=None, return_index=False):
        self.transform = transform
        self.dataset = dataset
        self.indexset = indexset
        self.class_split_indexes = {}
        self.label_list = []
        self.__len__()
        self._set_class_split_indexes()
        self.print_class_proportion()
        self.return_index = return_index

    def print_class_proportion(self):
        result = []
        keys = sorted(self.class_split_indexes.keys())
        for i in keys:
            result.append(len(self.class_split_indexes[i]))
        # print(np.asarray(result)/np.sum(result))

    def _set_class_split_indexes(self):
        for i in self.indexset:
            _, label = self.dataset[i]
            if label not in self.class_split_indexes.keys():
                self.class_split_indexes[label] = []
            self.class_split_indexes[label].append(i)
            self.label_list.append(label)

    def __len__(self):
        if self.indexset is None:
            self.indexset = [i for i in range(len(self.dataset))]
        return len(self.indexset)

    def __getitem__(self, indx):
        img, label = self.dataset[self.indexset[indx]]
        if self.transform:
            img = self.transform(img)
        if self.return_index:
            return self.indexset[indx], img, label
        else:
            return img, label

    def update_indexes(self, new_indexes):
        self.indexset = new_indexes

    def get_dataset_by_class(self, label):
        dst = BaseDataset()
        dst.data = self.dataset.data[self.class_split_indexes[label]]
        dst.targets = self.dataset.targets[self.class_split_indexes[label]]
        return DatasetWrapper(dst, transform=self.transform)

    def get_label_list(self):
        return self.label_list

    def update_transform(self, t):
        self.transform = t

    def get_dataset_by_indexes(self, indexes, has_transform=True):
        dst = BaseDataset(self.transform if has_transform else None)
        dst.data = self.dataset.data[indexes]
        dst.targets = self.dataset.targets[indexes]
        return dst

    def merge_from_dataset(self, dst:BaseDataset):
        self.dataset.data = np.concatenate((self.dataset.data, dst.data), axis=0)
        self.dataset.targets = np.concatenate((self.dataset.targets, dst.targets), axis=0)
        self.indexset = [i for i in range(len(self.dataset))]

    def delete_datas(self, indexes):
        self.dataset.data = np.delete(self.dataset.data, indexes, axis=0)
        self.dataset.targets = np.delete(self.dataset.targets, indexes, axis=0)
        self.indexset = [i for i in range(len(self.dataset))]

    def get_label_set(self):
        return self.class_split_indexes.keys()


def concatenate_datasets(dst_list):
    merged_dst = BaseDataset()
    data = []
    targets = []
    for dst in dst_list:
        data.append(dst.dataset.data)
        targets.append(dst.dataset.targets)
    merged_dst.data = np.concatenate(data, axis=0)
    merged_dst.targets = np.concatenate(targets, axis=0)
    return DatasetWrapper(merged_dst, transform=None)


def update_transform(dst:DatasetWrapper, t_type):
    if t_type == 'train':
        t = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        t = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    dst.update_transform(t)


def get_dataset(dst_name, split=None, index=None):

    if dst_name == 'im_cifar10':
        if split == 'train':
            dst = IMBALANCECIFAR10(root=PC.get_cifar10_dataset_path(),
                                   indexes_path=PC.get_cifar10_index_path()+'cifar10_train_indexes_1p.txt',
                     train=True, download=True)
        else:
            dst = IMBALANCECIFAR10(root=PC.get_cifar10_dataset_path(),
                                   indexes_path=PC.get_cifar10_index_path()+'cifar10_test_indexes_1p.txt',
                           train=False, download=True)
    elif dst_name == 'aug_cifar10':
        dst = AugmentedDataset(file_path=PC.get_cifar10_data_pool_path())
        print(len(dst))
    elif dst_name == 'coreset_cifar10':
        dst_indexes = np.loadtxt(PC.get_cifar10_index_path()+'cifar10_1p_valset_100p'+str(index)+'.txt')
        dst_indexes = dst_indexes.astype(dtype=int)
        augment_dst = DatasetWrapper(AugmentedDataset(file_path=PC.get_cifar10_data_pool_path()))
        print(len(augment_dst))
        dst = augment_dst.get_dataset_by_indexes(dst_indexes)
        del augment_dst
    elif dst_name == 'byorder_valset_cifar10':
        dst_indexes = np.loadtxt(PC.get_cifar10_index_path()+'byorder/cifar10_1p_valset_100p'+str(index)+'.txt')
        dst_indexes = dst_indexes.astype(dtype=int)
        augment_dst = DatasetWrapper(AugmentedDataset(file_path=PC.get_cifar10_data_pool_path()))
        dst = augment_dst.get_dataset_by_indexes(dst_indexes)
        del augment_dst
    dst = DatasetWrapper(dst)
    return dst


if __name__ == '__main__':
    generator = IMBALANCECIFAR10Generator(root=PC.get_cifar10_dataset_path(), train=True, download=True, imb_factor=0.01)
    generator.save_index_list(path=PC.get_cifar10_index_path()+'cifar10_train_indexes_1p.txt')

    generator = IMBALANCECIFAR10Generator(root=PC.get_cifar10_dataset_path(), train=False, download=True, imb_factor=0.01)
    generator.save_index_list(path=PC.get_cifar10_index_path()+'cifar10_test_indexes_1p.txt')
