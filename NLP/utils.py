
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