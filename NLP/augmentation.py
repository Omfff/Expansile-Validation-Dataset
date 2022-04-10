import tqdm
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from enum import Enum
from datasets import Dataset
from nltk.corpus import stopwords
import torch
from datasets import load_dataset
from nlpaug.util.file.download import DownloadUtil

# change to your local path
NLPAUG_MODEL_PATH = '/home/omf/.cache/nlpaug/model/'


def download():
    DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir=NLPAUG_MODEL_PATH)


class AugmentType(Enum):

    ContextualWordEmbsAugSubstitute = 0
    ContextualWordEmbsAugInsert=1
    SynonymAug=2
    # AntonymAug=3
    RandomWordAugDelete=4
    RandomWordAugCrop=5
    RandomWordAugSwap = 6
    BackTranslationAug=7
    AbstSummAug=8
    WordEmbsAugSubstitute = -2
    WordEmbsAugInsert = -1


class NLPAugment(object):
    def __init__(self, device='cpu', batch_size=8):
        self.device = device
        self.batch_size = batch_size
        self.stop_words = stopwords.words('english')

    def get_aug(self, aug_method, aug_prop):
        if aug_method == AugmentType.WordEmbsAugInsert:

            augmentor = naw.WordEmbsAug(model_type='fasttext',
                                               model_path=NLPAUG_MODEL_PATH+'wiki-news-300d-1M.vec',
                                               action='insert',
                                               aug_p=aug_prop, aug_max=None, stopwords=self.stop_words)
        elif aug_method == AugmentType.WordEmbsAugSubstitute:
            augmentor = naw.WordEmbsAug(model_type='fasttext',
                                                            model_path=NLPAUG_MODEL_PATH+'wiki-news-300d-1M.vec',
                                                            action='substitute',
                                                            aug_p=aug_prop, aug_max=None, stopwords=self.stop_words)
        elif aug_method == AugmentType.ContextualWordEmbsAugSubstitute:
            augmentor = naw.ContextualWordEmbsAug(model_path='bert-base-uncased',
                                      action="substitute", aug_p=aug_prop, aug_max=None, stopwords=self.stop_words, device=self.device,
                                      batch_size=self.batch_size) #"insert"
        elif aug_method == AugmentType.ContextualWordEmbsAugInsert:
            augmentor = naw.ContextualWordEmbsAug(model_path='bert-base-uncased',
                                      action="insert", aug_p=aug_prop, aug_max=None, stopwords=self.stop_words, device=self.device,
                                      batch_size=self.batch_size)  # "insert"
        elif aug_method == AugmentType.SynonymAug:
            augmentor = naw.SynonymAug(aug_src='wordnet', aug_p=aug_prop, aug_max=None, stopwords=self.stop_words)
        # self.aug_dict[AugmentType.AntonymAug] = naw.AntonymAug(aug_p=0.3, aug_max=None)
        elif aug_method == AugmentType.RandomWordAugDelete:
            augmentor = naw.RandomWordAug(action="delete", aug_p=aug_prop, aug_max=None, stopwords=self.stop_words)
        elif aug_method == AugmentType.RandomWordAugCrop:
            augmentor = naw.RandomWordAug(action="crop", aug_p=aug_prop, aug_max=None, stopwords=self.stop_words)
        elif aug_method == AugmentType.RandomWordAugSwap:
            augmentor = naw.RandomWordAug(action="swap", aug_p=aug_prop, aug_max=None, stopwords=self.stop_words)
        elif aug_method == AugmentType.BackTranslationAug:
            augmentor = naw.BackTranslationAug(
                from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en',
                device=self.device, batch_size=self.batch_size)
        elif aug_method == AugmentType.AbstSummAug:
            augmentor = nas.AbstSummAug(model_path='t5-base', device=self.device, batch_size=self.batch_size)
        return augmentor

    def generate_augmented_data(self, origin_dst:Dataset, save_path=None, multi_aug_p=False):
        label_set = list(set(origin_dst['label']))
        data_pool = {'text':[], 'label':[]}
        aug_prop_size = 2 if multi_aug_p else 1
        # bar = tqdm.tqdm(total=(len(AugmentType))*len(label_set)*aug_prop_size)
        bar = tqdm.tqdm(total=(len(AugmentType) - 2) * len(label_set) * aug_prop_size + 2 * len(label_set))
        bar_index = 0
        for at in AugmentType:
            print(at)
            if (not multi_aug_p) or at == AugmentType.BackTranslationAug or at == AugmentType.AbstSummAug:
                aug_prop_list = [0.3] # 0.3 or 0.5 for reuters
            elif multi_aug_p:
                aug_prop_list = [0.5, 0.7]
            else:
                raise Exception("Condition Error!")
            for aug_p in aug_prop_list:
                augmentor = self.get_aug(at, aug_prop=aug_p)
                for l in label_set:
                    class_dst = origin_dst.filter(lambda e: e['label']==l)
                    end_index = len(class_dst)
                    batch_start_index=0
                    while batch_start_index < end_index:
                        batch_end_index = batch_start_index + self.batch_size
                        if batch_end_index > end_index:
                            batch_end_index = end_index

                        curr_data = class_dst[batch_start_index:batch_end_index]['text']
                        try:
                            aug_data = augmentor.augment(curr_data)
                        except KeyboardInterrupt:
                            return
                        except:
                            batch_start_index += self.batch_size
                            continue

                        data_pool['text'].extend(aug_data)
                        data_pool['label'].extend([l]*len(aug_data))

                        batch_start_index += self.batch_size

                    bar.update(1)
                    bar_index += 1

                del augmentor
                torch.cuda.empty_cache()

        bar.close()

        import pandas as pd
        result = pd.DataFrame(data_pool)
        result.to_csv(save_path, index=False)
        if len(data_pool['label']) != ((len(AugmentType)-2)*aug_prop_size + 2)*len(origin_dst):
        # if len(data_pool['label']) != (len(AugmentType) * aug_prop_size) * len(origin_dst):
            raise Exception("The number of augmented data is error!")
        return result


if __name__ == '__main__':
    from utils import PathConfig
    PC = PathConfig()
    DATASET_PATH = PC.get_dataset_path()
    DATA_POOL_PATH = PC.get_data_pool_path()
    download()

    def generate_reuters_data_main():
        dst = load_dataset('csv', data_files={
            "train": DATASET_PATH + 'reuters_train.csv',
            "test": DATASET_PATH + 'reuters_test.csv'
        })
        nlp_auger = NLPAugment(device='cuda:0')
        nlp_auger.generate_augmented_data(dst['train'], save_path=DATA_POOL_PATH+'reuters/pool.csv',
                                          multi_aug_p=True)

    generate_reuters_data_main()