from transformers import AutoModelForSequenceClassification
from transformers import BertModel,BertTokenizer
from torch.utils.data import DataLoader
from enum import Enum
import torch
from ndb import NDB


class FeatureExtractorType(Enum):
    FineTune = 'fine-tune'
    PreTrain = 'pre-train'


class FeatureDistribution(object):
    def __init__(self, labels:list, device, feature_extractor_type, weight_path=None, dis_type=None):
        if feature_extractor_type == FeatureExtractorType.FineTune:
            self.feature_extractor = AutoModelForSequenceClassification.from_pretrained(weight_path, output_hidden_states=True)
        elif feature_extractor_type == FeatureExtractorType.PreTrain:
            self.feature_extractor = BertModel.from_pretrained('bert-base-uncased')
        self.fe_type = feature_extractor_type
        self.feature_extractor.to(device)
        self.device = device
        self.labels = labels
        self.dis_type = dis_type
        if dis_type == 'NDB':
            self.dis_calculator = {}
            for l in labels:
                self.dis_calculator[l] = None

    def cal_class_distribution(self, class_x_set, bs=64):
        loader = DataLoader(class_x_set, batch_size=bs)
        feature_list = []
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.feature_extractor(**batch)
                if self.fe_type == FeatureExtractorType.PreTrain:
                    features = torch.mean(outputs.last_hidden_state, dim=1).unsqueeze(dim=0)
                elif self.fe_type == FeatureExtractorType.FineTune:
                    # features = torch.mean(outputs.hidden_states[-2], dim=1).unsqueeze(dim=0)
                    # hidden_states(num_layer, batch_size, max_seq, feature_dim)
                    features = outputs.hidden_states[-2][:, 0, :].detach().unsqueeze(dim=0)
                feature_list.append(features)
        features = torch.cat(feature_list, dim=1)
        if len(class_x_set) != features.shape[1]:
            raise Exception('cal_class_distribution error!')
        return features

    def cal_distribution_diff_in_class(self, f1, f2, label=None):
        if self.dis_type == 'NDB':
            if self.dis_calculator[label] is None:
                self.dis_calculator[label] = NDB(training_data=f2.squeeze(dim=0).cpu().detach().numpy(), number_of_bins=10, whitening=True)
            return self.dis_calculator[label].evaluate(f1.squeeze(dim=0).cpu().detach().numpy())['JS']
        else:
            return self.dis_calculator(f1, f2, bidirectional=True).detach().cpu().item()

    def cal_distribution(self, x, y, batch_size=64):
        x.set_format("torch")
        feature_distri_dict = {l: None for l in self.labels}
        for label in self.labels:
            class_x = x.filter(lambda e, i: y[i]==label, with_indices=True)
            feature_distri_dict[label] = self.cal_class_distribution(class_x, batch_size)
        return feature_distri_dict

    def cal_distribution_diff(self, f1, f2):
        distri_dis_dict = {}
        for label in self.labels:
            distri_dis_dict[label] = self.cal_distribution_diff_in_class(f1[label], f2[label], label)
        return distri_dis_dict

    def cal_distribution_diff_for_two_set(self, x1, y1, x2, y2):
        f1 = self.cal_distribution(x1, y1)
        f2 = self.cal_distribution(x2, y2)
        dis = self.cal_distribution_diff(f1, f2)
        return dis


def ndb_test():
    from ndb import NDB
    from datasets import load_dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dst = load_dataset('csv', data_files={
        "train": 'dataset/reuters_train.csv',
        "test": 'dataset/reuters_test.csv'
    })
    from utils import split_train_val
    def tokenize_function(dst):
        return tokenizer(dst['text'], padding=True, truncation=True, max_length=None)
    whole_train_dst = dst["train"]
    train_dst = whole_train_dst.select([i for i in range(32)]).remove_columns('label')
    train_dst = train_dst.map(tokenize_function, batched=True)
    train_dst.set_format('torch')
    shuffle_train_dst = train_dst.shuffle()
    feature_extractor0 = BertModel.from_pretrained('bert-base-uncased')
    outputs0 = torch.mean(feature_extractor0(**{'attention_mask':train_dst['attention_mask'], 'input_ids':train_dst['input_ids'],  'token_type_ids':train_dst['token_type_ids']}).last_hidden_state, dim=1)
    outputs1 = torch.mean(feature_extractor0(**{'attention_mask':shuffle_train_dst['attention_mask'], 'input_ids':shuffle_train_dst['input_ids'],  'token_type_ids':shuffle_train_dst['token_type_ids']}).last_hidden_state, dim=1)
    ndb = NDB(training_data=outputs0.detach().numpy(), number_of_bins=3, whitening=True)
    ndb.evaluate(outputs0.detach().numpy(), model_label='shuffle')


if __name__ == '__main__':
    ndb_test()
