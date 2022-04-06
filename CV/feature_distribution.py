from torch.utils.data import DataLoader
from chamferdist import ChamferDistance
from enum import Enum
import torch
from ndb import NDB
from imbalanced_dataset import DatasetWrapper


class FeatureExtractor(object):
    def __init__(self, feature_extractor_type, model, device):
        self.feature_extractor = model
        self.feature_extractor.eval()
        self.feature_extractor.to(device)
        self.fe_type = feature_extractor_type
        self.device = device

    def get_features(self, datas, is_logits=False):
        with torch.no_grad():
            datas = datas.to(self.device)
            logits, features = self.feature_extractor(datas)
            if is_logits:
                return logits.unsqueeze(dim=0)
            else:
                return features.unsqueeze(dim=0)


class FeatureExtractorType(Enum):
    FineTune = 'fine-tune'
    PreTrain = 'pre-train'


class FeatureDistribution(object):
    """ Measure the feature distribution of a set and the distance between two sets"""
    def __init__(self, labels:list, feature_extractor:FeatureExtractor, dis_type=None):
        """
        :param labels: label(class) list
        :param feature_extractor: feature extractor(model)
        :param dis_type: 'NDB'|'CD', representation for the feature distribution of a set
        """
        self.feature_extractor = feature_extractor
        self.labels = labels
        self.dis_type = dis_type
        if dis_type == 'NDB':
            self.dis_calculator = {}
            for l in labels:
                self.dis_calculator[l] = None
        elif dis_type == 'CD':
            self.dis_calculator = ChamferDistance()

    def cal_class_distribution(self, class_x_set, bs=64):
        loader = DataLoader(class_x_set, batch_size=bs)
        feature_list = []
        for datas, _ in loader:
            features = self.feature_extractor.get_features(datas)
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

    def cal_distribution(self, x:DatasetWrapper, y, batch_size=64):
        feature_distri_dict = {l: None for l in self.labels}
        for label in self.labels:
            class_x = x.get_dataset_by_class(label)
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

