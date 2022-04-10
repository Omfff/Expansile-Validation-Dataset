from nltk.corpus import reuters
from transformers import AutoTokenizer
from datasets import load_metric, load_dataset


def post_process(dst, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    def tokenize_function(dst):
        return tokenizer(dst['text'], padding='max_length', truncation=True, max_length=512)
        # return tokenizer(dst['text'], padding=True, truncation=True, max_length=None)
    dst = dst.map(tokenize_function, batched=True)
    dst = dst.remove_columns(["text"])
    dst = dst.rename_column("label", "labels")
    return dst


def decode_to_sentence(dst, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    def decode(data):
        return tokenizer.decode(data, skip_special_tokens=True)
    dst = dst.map(lambda example: {'text': decode(example['input_ids'])} , num_proc=4)
    return dst


def get_dataset(dst_name, model_name, seed=None):
    if dst_name == 'wheat_corn_reuters':
        # train_dst, test_dst = get_dataset_from_nltk()
        # train_dst = datasets.Dataset.from_pandas(train_set).remove_columns(['__index_level_0__'])
        # test_dst = datasets.Dataset.from_pandas(test_set).remove_columns(['__index_level_0__'])
        dst = load_dataset('csv', data_files={
            "train": 'dataset/reuters_train.csv',
            "test": 'dataset/reuters_test.csv'
        })
        train_dst, test_dst = dst['train'], dst['test']
        train_dst = post_process(train_dst, model_name)
        test_dst = post_process(test_dst, model_name)
        print('train shape', train_dst.shape)
        print('test shape', test_dst.shape)
        return train_dst, test_dst
    elif dst_name == 'wheat_corn_reuters_nosplit':
        dst = load_dataset('csv', data_files={
            "train": 'dataset/reuters_coling2018.csv',
        })
        dst = dst['train']
        dst = post_process(dst, model_name)
        print('train shape', dst.shape)
        return dst
    elif dst_name == 'augmented_wheat_corn_reuters':
        pool_set = load_dataset('csv', data_files='./data_pool/reuters/version1/pool.csv')['train']
        dst = post_process(pool_set, model_name)
        dst.set_format('torch')
        return dst
    elif dst_name == 'augmented_wheat_corn_reuters_valset':
        import numpy as np
        dst_indexes = np.loadtxt('data_pool/reuters/version1/reuters_wheat_corn_valset_100p' + str(seed) + '.txt')
        dst_indexes = dst_indexes.astype(dtype=int)
        augment_dst = load_dataset('csv', data_files='./data_pool/reuters/version1/pool.csv')['train']
        dst = augment_dst.select(dst_indexes.tolist())
        del augment_dst
        dst = post_process(dst, model_name)
        dst.set_format('torch')
        return dst
    elif dst_name == 'augmented_wheat_corn_reuters_valset_byorder':
        import numpy as np
        dst_indexes = np.loadtxt('data_pool/reuters/byorder/reuters_wheat_corn_valset_100p' + str(seed) + '.txt')
        dst_indexes = dst_indexes.astype(dtype=int)
        augment_dst = load_dataset('csv', data_files='./data_pool/reuters/version1/pool.csv')['train']
        dst = augment_dst.select(dst_indexes.tolist())
        del augment_dst
        dst = post_process(dst, model_name)
        dst.set_format('torch')
        return dst
    elif dst_name == 'imdb':
        dst = load_dataset('csv', data_files={
            "train": '/ssd1/omf/datasets/imdb/imdb_train.csv',
            "test": '/ssd1/omf/datasets/imdb/imdb_test.csv'
        })
        train_dst, test_dst = dst['train'], dst['test']
        train_dst = post_process(train_dst, model_name)
        test_dst = post_process(test_dst, model_name)
        print('train shape', train_dst.shape)
        print('test shape', test_dst.shape)
        return train_dst, test_dst
    elif dst_name == 'augmented_imdb':
        pool_set = load_dataset('csv', data_files='./data_pool/imdb/pool.csv')['train']
        dst = post_process(pool_set, model_name)
        dst.set_format('torch')
        return dst
    elif dst_name == 'augmented_imdb_valset':
        import numpy as np
        dst_indexes = np.loadtxt('data_pool/imdb/imdb_valset_100p' + str(seed) + '.txt')
        dst_indexes = dst_indexes.astype(dtype=int)
        augment_dst = load_dataset('csv', data_files='./data_pool/imdb/pool.csv')['train']
        dst = augment_dst.select(dst_indexes.tolist())
        del augment_dst
        dst = post_process(dst, model_name)
        dst.set_format('torch')
        return dst
    elif dst_name == 'augmented_imdb_valset_byorder':
        import numpy as np
        dst_indexes = np.loadtxt('data_pool/imdb/byorder/imdb_valset_100p' + str(seed) + '.txt')
        dst_indexes = dst_indexes.astype(dtype=int)
        augment_dst = load_dataset('csv', data_files='./data_pool/imdb/pool.csv')['train']
        dst = augment_dst.select(dst_indexes.tolist())
        del augment_dst
        dst = post_process(dst, model_name)
        dst.set_format('torch')
        return dst
    elif dst_name == 'newsgroups':
        dst = load_dataset('csv', data_files={
            "train": '/ssd1/omf/datasets/newsgroups/newsgroups_train.csv',
            "test": '/ssd1/omf/datasets/newsgroups/newsgroups_test.csv'
        })
        train_dst, test_dst = dst['train'], dst['test']
        train_dst = post_process(train_dst, model_name)
        test_dst = post_process(test_dst, model_name)
        print('train shape', train_dst.shape)
        print('test shape', test_dst.shape)
        return train_dst, test_dst


if __name__ == '__main__':
    train, test = get_dataset('wheat_corn_reuters', model_name = "distilbert-base-uncased")
    print(len(train) + len(test))
    print(len(train.filter(lambda example: example['labels'] == 1))+ len(test.filter(lambda example: example['labels'] == 1)))