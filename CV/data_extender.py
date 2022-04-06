import pandas as pd
import tqdm
import copy
import math
import torch
import numpy as np
from feature_distribution import FeatureDistribution
from imbalanced_dataset import DatasetWrapper, get_dataset, concatenate_datasets, update_transform


def get_sample_num_by_class(dst):
    return {l: len(dst[l]) for l in dst}


class SampleAddNumManager(object):
    def __init__(self, init_sample_num, add_num_per_iter, decay_rate, decay_method=None, stage=None):
        """

        :param decay_method: None | "multi_stage" | "triggered"
        """
        self.init_sample_num = init_sample_num
        self.add_num_per_iter = add_num_per_iter
        self.add_num_decay_rate = decay_rate
        self.decay_method = decay_method
        if decay_method == "multi_stage":
            self.decay_stage = stage

        self.curr_decay_rate = 1
        self.curr_iter = 0

        print('init sample num', init_sample_num)

    def get_curr_iter_add_num(self, is_prev_unfinished=False):
        if self.decay_method == "multi_stage" and self.curr_iter in self.decay_stage:
            self.update_curr_decay_rate()
        if self.decay_method == 'triggered' and is_prev_unfinished:
            self.update_curr_decay_rate()

        current_add_num = copy.deepcopy(self.add_num_per_iter)
        for l in self.add_num_per_iter.keys():
            current_add_num[l] = math.ceil(current_add_num[l] * self.curr_decay_rate)
        self.curr_iter += 1
        print("curr add sample num", current_add_num)
        return current_add_num

    def update_curr_decay_rate(self):
        self.curr_decay_rate = self.curr_decay_rate * self.add_num_decay_rate

    def get_last_iter_add_num(self, curr_sample_num):
        max_expand_rate = 1
        for l in self.init_sample_num.keys():
            max_expand_rate = max_expand_rate if curr_sample_num[l]/self.init_sample_num[l] < max_expand_rate \
                else curr_sample_num[l]/self.init_sample_num[l]

        current_add_num = {}
        for l in self.init_sample_num.keys():
            current_add_num[l] = math.ceil(self.init_sample_num[l] * max_expand_rate - curr_sample_num[l])
            current_add_num[l] = current_add_num[l] if current_add_num[l] >0 else 1

        print("last add sample num", current_add_num)
        return current_add_num


class DataExtender(object):
    def __init__(self, whole_train_set, init_train_set, init_val_set, fd: FeatureDistribution = None, iter_num=None,
                 add_ratio_per_iter=None, diff_threshold_ratio=0.1, early_stop_threshold=3, try_num_limits=300,
                 add_num_decay_rate=0.5, add_num_decay_method=None, add_num_decay_stage=None,
                 random_seed=0):
        """ Expand validation set iteratively

        :param whole_train_set: source set
        :param init_train_set: train set(after splitting)
        :param init_val_set: initial validation set
        :param fd: object that could calculate feature distribution of dataset
        :param iter_num: total iteration rounds
        :param add_ratio_per_iter: proportion of samples added per iteration
        :param diff_threshold_ratio: The threshold for each category is equal to diff_threshold_ratio*initial-fdd
                (feature distribution distance)
        :param early_stop_threshold: maximum number of consecutive rounds with tries reach the limitation
        :param try_num_limits: max tries per iteration
        :param add_num_decay_rate: the shrinking rate of add number when tries reach the try_num_limits
        :param add_num_decay_method: 'triggered'|'step'
        :param add_num_decay_stage: add_num decaying rounds if add_num_decay_method is 'step' else None
        :param random_seed:
        """

        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.whole_train_set = whole_train_set
        self.train_set = init_train_set
        self.init_val_set = init_val_set

        self.iter_num = iter_num
        self.add_ratio = add_ratio_per_iter
        self.data_pool = {}

        self.fd_calculator = fd
        self.whole_train_set_fd = self.fd_calculator.cal_distribution(whole_train_set, None)
        self.init_val_fd = self.fd_calculator.cal_distribution(init_val_set, None)
        self.init_fd_diff = self.fd_calculator.cal_distribution_diff(self.init_val_fd, self.whole_train_set_fd)

        self.label_set = list(self.init_fd_diff.keys())
        self.class_split_val_dst = self.split_set_by_class(self.init_val_set)
        self.add_num_per_iter = self.cal_add_num_per_iter()

        # calculate fdd threshold for each category
        self.threshold = self.__init_threshold(diff_threshold_ratio)

        self.early_stop_t = early_stop_threshold
        self.try_num_limits = try_num_limits

        self.an_manager = SampleAddNumManager(get_sample_num_by_class(self.class_split_val_dst),
                                              self.add_num_per_iter, decay_rate=add_num_decay_rate, decay_method=add_num_decay_method,
                                              stage=add_num_decay_stage)

        print('early_stop_t %d try_num_limits %d'%(self.early_stop_t, self.try_num_limits))
        print('threshold', self.threshold)

    def split_set_by_class(self, dataset:DatasetWrapper):
        class_split_dset = {}
        for l in self.label_set:
            class_split_dset[l] = dataset.get_dataset_by_class(l)
        return class_split_dset

    def cal_add_num_per_iter(self):
        result = {}
        for label, samples in self.class_split_val_dst.items():
            # the only difference with origin file
            result[label] = math.ceil(len(samples) * self.add_ratio) #int
        return result

    def generate_data_to_pool(self, name):
        """ Load auxiliary dataset
        """
        dst = get_dataset(name)
        update_transform(dst, t_type='test')
        self.data_pool = self.split_set_by_class(dst)

    def check_label_distribution_change(self, dst1, dst2):
        """ Check degree of reduction in feature distribution distance compared to pre-expansion
        """
        change_rate = {}
        for l in dst1.keys():
            change_rate[l] = len(dst2[l])/len(dst1[l])
        print("label_distribution_change", change_rate)

    def merge_set(self, dst):
        """ Merge a set(Dataset) dict to a set

        :param dst: {class: set, ...}
        :return: merged dataset
        """
        merged_set =[]
        for label in self.label_set:
            merged_set.append(dst[label])
        merged_set = concatenate_datasets(merged_set)
        update_transform(merged_set, t_type='test')
        return merged_set

    def __init_threshold(self, diff_threshold_ratio):
        """ Set initial threshold for each category to diff_threshold_ratio*initial-feature-distribution-distance
        """
        total_diff = np.sum(list(self.init_fd_diff.values()))
        return {l: self.init_fd_diff[l] * (diff_threshold_ratio*(self.init_fd_diff[l]/total_diff)) for l in self.label_set}

    def decay_threshold(self, ratio=2):
        for l in self.label_set:
            self.threshold[l] = self.threshold[l]/ratio

    def run(self, ignore_feature_distance=False):
        """ Main func for iterative expanding

        :param ignore_feature_distance: if True, random sample will be token in each iteration
        :return: Expanded validation set
        """
        # the number of consecutive rounds with tries reach the limitation
        unfinished_tolerance = 0
        print(self.init_fd_diff)

        val_set = copy.deepcopy(self.class_split_val_dst)
        temp_val_set = copy.deepcopy(self.class_split_val_dst)
        temp_val_set_fd = self.fd_calculator.cal_distribution(self.init_val_set, None)
        for j in range(self.iter_num):
            # reach the limitation
            if unfinished_tolerance >= self.early_stop_t:
                break

            if j == self.iter_num - 1:
                # the added size should ensure that the proportion of total additions is consistent across categories
                current_add_num = self.an_manager.get_last_iter_add_num(get_sample_num_by_class(val_set))
            else:
                # maintain or decay the amount added according to the tries in the previous iteration
                current_add_num = self.an_manager.get_curr_iter_add_num(is_prev_unfinished=(unfinished_tolerance > 0))

            if unfinished_tolerance > 0:
                self.decay_threshold(ratio=4)

            curr_val_set_fd = temp_val_set_fd

            for label_index in range(len(self.label_set)):
                print(('='*10+'iter num %d class %d running'+'='*10) % (j, self.label_set[label_index]))
                if ignore_feature_distance:
                    # random sample
                    val_set, curr_val_set_fd , is_finished, total_tries = self.random_sample_for_one_class(val_set,
                                        curr_val_set_fd, label_index, current_add_num[self.label_set[label_index]], j)
                else:
                    val_set, curr_val_set_fd , is_finished, total_tries = self.run_one_iter_one_class(val_set, curr_val_set_fd,
                                                label_index, current_add_num[self.label_set[label_index]], j)

                if not is_finished:
                    # reset val set to last status
                    val_set = temp_val_set
                    unfinished_tolerance += 1
                    print('unfinished_tolerance + 1 = %d'%(unfinished_tolerance))
                    break

                if label_index == len(self.label_set) - 1 and is_finished:
                    unfinished_tolerance = 0
                    # backup val set and fd
                    temp_val_set = copy.deepcopy(val_set)
                    temp_val_set_fd = copy.deepcopy(curr_val_set_fd)

        self.check_label_distribution_change(self.class_split_val_dst, val_set)
        val_set = self.merge_set(val_set)

        return self.train_set, val_set

    def run_one_iter_one_class(self, val_set, curr_val_set_fd, label_index, add_num, iter_index, sample_method='RS'):
        """ The expansion of one category in a single iteration
        """
        total_tries = 0
        curr_add_num = add_num
        remain_add_num = add_num
        try_num = 0
        label = self.label_set[label_index]
        curr_threshold = self.threshold[label]
        while remain_add_num != 0:
            # previous feature distribution distance
            prev_class_fd = curr_val_set_fd[label]
            prev_distri_diff = self.fd_calculator. \
                cal_distribution_diff_in_class(prev_class_fd, self.whole_train_set_fd[label], label)

            if sample_method == 'RS':
                sample_indexes = np.random.randint(0, len(self.data_pool[label]), size=curr_add_num).tolist()
                samples = self.data_pool[label].get_dataset_by_indexes(sample_indexes)

            # current feature distribution distance
            new_samples_fd = self.fd_calculator.cal_class_distribution(samples)
            curr_class_fd = torch.cat([prev_class_fd, new_samples_fd], dim=1)
            curr_distri_diff = self.fd_calculator. \
                cal_distribution_diff_in_class(curr_class_fd, self.whole_train_set_fd[label], label)

            try_num += 1
            if prev_distri_diff - curr_distri_diff > curr_threshold:
                if sample_method == 'RS':
                    self.data_pool[label].delete_datas(sample_indexes)
                val_set[label].merge_from_dataset(samples)
                curr_val_set_fd[label] = curr_class_fd
                print('iter num %d class %d try times %d add num %d' % (iter_index, label, try_num, curr_add_num))
                print('iter num %d class %d current distribution diff %f' % (iter_index, label, curr_distri_diff))
                remain_add_num = remain_add_num - curr_add_num
                curr_add_num = remain_add_num if remain_add_num < curr_add_num else curr_add_num
                total_tries +=try_num
                try_num = 0
            else:
                if try_num > self.try_num_limits:
                    if curr_add_num == 1:
                        print('iter num %d class %d try times reach limitation' % (iter_index, label))
                        break
                    else:
                        # shrink the addition and decay the threshold
                        curr_add_num = math.ceil(curr_add_num/2)
                        curr_threshold = curr_threshold/10
                        total_tries += try_num
                        try_num = 0

        is_finished = True if remain_add_num == 0 else False

        return val_set, curr_val_set_fd , is_finished, total_tries

    def random_sample_for_one_class(self, val_set, curr_val_set_fd, label_index, add_num, iter_index):
        label = self.label_set[label_index]
        prev_class_fd = curr_val_set_fd[label]

        sample_indexes = np.random.randint(0, len(self.data_pool[label]), size=add_num).tolist()
        samples = self.data_pool[label].get_dataset_by_indexes(sample_indexes)

        new_samples_fd = self.fd_calculator.cal_class_distribution(samples)
        curr_class_fd = torch.cat([prev_class_fd, new_samples_fd], dim=1)
        curr_distri_diff = self.fd_calculator. \
            cal_distribution_diff_in_class(curr_class_fd, self.whole_train_set_fd[label], label)

        self.data_pool[label].delete_datas(sample_indexes)
        val_set[label].merge_from_dataset(samples)
        curr_val_set_fd[label] = curr_class_fd

        print('iter num %d class %d add num %d' % (iter_index, label, add_num))
        print('iter num %d class %d current distribution diff %f' % (iter_index, label, curr_distri_diff))

        is_finished = True
        try_num = 1

        return val_set, curr_val_set_fd, is_finished, try_num





