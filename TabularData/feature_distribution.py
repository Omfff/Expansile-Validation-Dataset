from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon


def plot_embedding(data, label, title, legend = None, cn = None, selected_index=None):
    if legend is None:
        legend = [i for i in range(0,cn)]

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    scatter = plt.scatter(data[:, 0], data[:, 1], c=label, s=4)
    select_point = plt.scatter(data[selected_index[0], 0], data[selected_index[0], 1],
                               marker='o', c='',edgecolors='g', s=20)
    # for i in range(data.shape[0]):
    #     plt.scatter(data[i, 0], data[i, 1], c=plt.cm.Set1(label[i] / 10.))
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.legend(handles = scatter.legend_elements()[0],labels=legend, loc='best')
    return fig


def tsne_vis(data, labels, val_index):
    tsne_result = TSNE(n_components=2, perplexity=10, init='pca', n_iter=5000).fit_transform(data)
    fig = plot_embedding(tsne_result, labels, title='', cn=2, selected_index=val_index)
    plt.show()


class FeatureType(object):
    def __init__(self, numerical_f, oridinal_f, categorical_f):
        self.numerical_cols = numerical_f
        self.oridinal_cols = oridinal_f
        self.categorical_cols = categorical_f
        self.discrete_cols = oridinal_f+categorical_f

    def get_numerical_cols(self):
        return self.numerical_cols

    def get_oridinal_cols(self):
        return self.oridinal_cols

    def get_categorical_cols(self):
        return self.categorical_cols

    def get_discrete_cols(self):
        return self.discrete_cols


class FeatureDistribution(object):
    def __init__(self, x_set:pd.DataFrame, y_set, labels:list, discrete_col, numerical_col, k=5):
        # self.y_set = y_set.reset_index(drop=True)
        # self.x_set = x_set.reset_index(drop=True)
        self.y_set = y_set
        self.x_set = x_set
        self.labels = labels
        self.discrete_col = discrete_col
        self.numerical_col = numerical_col
        self.cate_dict = {l:{} for l in labels}
        self.bins_dict = {l:{} for l in labels}
        self.bins_num = 10
        self.k = k

    def check_correctness(self, f1, f2):
        for label in self.labels:
            for col in f1[label].keys():
                if sum(f1[label][col]) != sum(f2[label][col]):
                    print('error', col, f1[label][col], f2[label][col])

    def add_in_class_feature_distribution(self, f1, f2):
        result_f = {}
        for col in f1.keys():
            result_f[col] = np.asarray(f1[col],dtype=int) + np.asarray(f2[col], dtype=int)
            result_f[col] = result_f[col].tolist()
        return result_f

    def normalize_feature_frequence_to_distribution(self, ff):
        result = {}
        for col in ff.keys():
            temp = np.asarray(ff[col])
            result[col] = (temp/np.sum(temp)).tolist()
        return result

    def cal_class_distribution(self, class_x_set, label, normalize=True):
        # class_x_set = x[y == label]
        feature_distri_dict = {}

        for cc in self.discrete_col:
            if cc not in self.cate_dict[label].keys():
                f_distribution = class_x_set[cc].value_counts(sort=False, normalize=normalize)
                f_distribution = f_distribution.sort_index()
                cate_list = list(f_distribution.index)
                self.cate_dict[label][cc] = cate_list
                self.cate_dict[label][cc].append(cate_list[-1] + 1)
                self.cate_dict[label][cc] = [c - 0.5 for c in self.cate_dict[label][cc]]
            else:
                f_distribution = pd.value_counts(
                    pd.cut(class_x_set[cc], self.cate_dict[label][cc],
                           labels=[i + 1 for i in range(len(self.cate_dict[label][cc]) - 1)]),
                    sort=False, normalize=normalize)
            feature_distri_dict[cc] = f_distribution.values.tolist()

        for nc in self.numerical_col:
            if nc not in self.bins_dict[label].keys():
                f_distribution = class_x_set[nc].value_counts(bins=self.bins_num, sort=False, normalize=normalize)
                bins = list(f_distribution.index)
                left_b = [interval.left for interval in bins]
                right_b = [interval.right for interval in bins]
                bins = sorted(list(set(left_b) | set(right_b)))
                bins[-1] = bins[-1] + 1
                self.bins_dict[label][nc] = bins
            else:
                f_distribution = pd.value_counts(pd.cut(class_x_set[nc], self.bins_dict[label][nc]), sort=False,
                                                 normalize=normalize)
            feature_distri_dict[nc] = f_distribution.values.tolist()
        return feature_distri_dict

    def cal_distribution_diff_in_class(self, f1, f2):
        """
        cal feature distribution diff between two in-class distribution
        :param f1:
        :param f2:
        :return:
        """
        f1_distri_list = []
        f2_distri_list = []
        for col in f1.keys():
            f1_distri_list.extend(f1[col])
            f2_distri_list.extend(f2[col])
        dis = wasserstein_distance(f1_distri_list, f2_distri_list)
        # dis = jensenshannon(f1_distri_list, f2_distri_list)
        return dis

    def cal_distribution(self, x, y, normalize=True):
        feature_distri_dict = {l: {} for l in self.labels}
        for label in self.labels:
            class_x_set = x[y == label]
            feature_distri_dict[label] = self.cal_class_distribution(class_x_set, label, normalize=normalize)
        return feature_distri_dict

    def cal_distribution_diff(self, f1, f2):
        distri_dis_dict = {}
        for label in self.labels:
            distri_dis_dict[label] = self.cal_distribution_diff_in_class(f1[label], f2[label])
        return distri_dis_dict

    def run(self):
        """
         cal the average feature distribution diff between valset and whole trainset under StratifiedKFold method
        :return:
        """
        global_distribution = self.cal_distribution(self.x_set, self.y_set)
        # global_distribution2 = self.cal_distribution(self.x_set, self.y_set)
        # self.check_correctness(global_distribution, global_distribution2)
        kfold = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=10)
        result = []
        for train_ind, val_ind in kfold.split(self.x_set, self.y_set):
            distribution = self.cal_distribution(self.x_set.loc[val_ind], self.y_set.loc[val_ind])
            distance_dict = self.cal_distribution_diff(global_distribution, distribution)
            result.append(distance_dict)

        mean_distribution_dis = {l: [] for l in self.labels}
        for i in range(len(result)):
            for l in self.labels:
                mean_distribution_dis[l].append(result[i][l])
        print(mean_distribution_dis)
        avg_dis = {l: [] for l in self.labels}
        for l in self.labels:
            avg_dis[l].append(np.mean(mean_distribution_dis[l]))
            avg_dis[l].append(np.std(mean_distribution_dis[l]))
        print(avg_dis)
        return avg_dis

