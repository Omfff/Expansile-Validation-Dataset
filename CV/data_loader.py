from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split, StratifiedKFold


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


def get_dataloader(args, train_val_set, test_dataset, usage='train'):

    if usage == 'train':
        train_loader = DataLoader(train_val_set[0], batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  pin_memory=True)

        val_loader = DataLoader(train_val_set[1], batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                  pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True)

        return train_loader, val_loader, test_loader


class TestArgs(object):
    def __init__(self):
        self.batch_size = 64
        self.workers = 8
        self.data_index_save_dir = './data_pool/'
        self.iteration = 0


if __name__ == '__main__':
    args = TestArgs()
    loader = get_dataloader(args, usage='select')
    for i,(indexes, images, targets) in enumerate(loader):
        print(images.shape)
        break


