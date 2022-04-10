from nltk.corpus import reuters
import pandas as pd
from utils import PathConfig


DATASET_SAVE_PATH = PathConfig().get_dataset_path()


def generate_dataset_from_nltk():
    """
        Original code from https://github.com/henrymoss/COLING2018/blob/master/non-interactive/Reuters/REUTERSSVM_tuner.py
    """
    wheat_docs = reuters.fileids("wheat")
    corn_docs = reuters.fileids("corn")
    wheat_train = list(filter(lambda doc: doc.startswith("train"), wheat_docs))
    wheat_test = list(filter(lambda doc: doc.startswith("test"), wheat_docs))
    corn_train = list(filter(lambda doc: doc.startswith("train"), corn_docs))
    corn_test = list(filter(lambda doc: doc.startswith("test"), corn_docs))
    training_index = wheat_train + corn_train
    test_index = wheat_test + corn_test

    # prepare data for wheat vs not wheat case
    def convert_to_df(index):
        from sklearn.utils import shuffle
        mix_num = 0
        text = []
        clas = []
        classname = ["corn", "wheat"]
        for i in index:
            if "wheat" in reuters.categories(i):
                clas.append(1)
            else:
                clas.append(0)
            text.append(reuters.raw(i))
        # store in dataframe
        data = pd.DataFrame(clas, columns=['label'])
        data["text"] = text
        data = shuffle(data, random_state=0)
        print("mix_num ",mix_num)
        return data

    train_set = convert_to_df(training_index)
    test_set = convert_to_df(test_index)
    train_set.to_csv('dataset/reuters_train.csv', index=False)
    test_set.to_csv('dataset/reuters_test.csv', index=False)
    return train_set, test_set


if __name__ == '__main__':
    generate_dataset_from_nltk()