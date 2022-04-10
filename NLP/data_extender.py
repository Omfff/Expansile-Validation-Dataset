from feature_distribution import FeatureDistribution
from datasets import Dataset
import copy
import math
from datasets import load_dataset, concatenate_datasets
import torch
import numpy as np


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

        self.random_seed = random_seed
        np.random.seed(random_seed)
        print("data extender seed: %d"%(random_seed))

        self.whole_train_set = whole_train_set
        self.train_set = init_train_set
        self.init_val_set = init_val_set

        self.iter_num = iter_num
        self.add_ratio = add_ratio_per_iter
        self.data_pool = {}

        self.fd_calculator = fd
        self.whole_train_set_fd = self.fd_calculator.cal_distribution(whole_train_set.remove_columns(['labels']),
                                                              whole_train_set['labels'])
        self.init_val_fd = self.fd_calculator.cal_distribution(init_val_set.remove_columns(['labels']),
                                                               init_val_set['labels'])
        self.init_fd_diff = self.fd_calculator.cal_distribution_diff(self.init_val_fd, self.whole_train_set_fd)

        self.label_set = list(self.init_fd_diff.keys())
        self.class_split_val_dst = self.split_set_by_class(self.init_val_set)
        self.add_num_per_iter = self.cal_add_num_per_iter()

        self.threshold = self.__init_threshold(diff_threshold_ratio)

        self.early_stop_t = early_stop_threshold
        self.try_num_limits = try_num_limits

        self.an_manager = SampleAddNumManager(get_sample_num_by_class(self.class_split_val_dst),
                                              self.add_num_per_iter, decay_rate=add_num_decay_rate, decay_method=add_num_decay_method,
                                              stage=add_num_decay_stage)

    def split_set_by_class(self, dataset:Dataset):
        class_split_dset = {}
        for l in self.label_set:
            class_split_dset[l] = dataset.filter(lambda example: example['labels'] == l)
        return class_split_dset

    def cal_add_num_per_iter(self):
        result = {}
        for label, samples in self.class_split_val_dst.items():
            result[label] = int(len(samples) * self.add_ratio)
        return result

    def generate_data_to_pool(self, cache_files, process_func, *args):
        if cache_files is not None:
            pool_set = load_dataset('csv', data_files=cache_files)['train']
            for l in self.label_set:
                self.data_pool[l] = process_func(pool_set.filter(lambda e:e['label']==l), *args)
                self.data_pool[l].set_format('torch')
        else:
            pass

    def check_label_distribution_change(self, dst1, dst2):
        change_rate = {}
        for l in dst1.keys():
            change_rate[l] = len(dst2[l])/len(dst1[l])
        print("label_distribution_change", change_rate)

    def merge_set(self, dst):
        merged_set =[]
        for label in self.label_set:
            merged_set.append(dst[label])
        merged_set = concatenate_datasets(merged_set, axis=0)
        return merged_set

    def __init_threshold(self, diff_threshold_ratio):
        total_diff = np.sum(list(self.init_fd_diff.values()))
        return {l: self.init_fd_diff[l] * (diff_threshold_ratio*(self.init_fd_diff[l]/total_diff)) for l in self.label_set}

    def decay_threshold(self, ratio=2):
        for l in self.label_set:
            self.threshold[l] = self.threshold[l]/ratio

    def run(self, sample_method='RS', ignore_feature_distance=False):
        unfinished_tolerance = 0
        print(self.init_fd_diff)

        val_set = copy.deepcopy(self.class_split_val_dst)
        temp_val_set = copy.deepcopy(self.class_split_val_dst)
        temp_val_set_fd = self.fd_calculator.cal_distribution(self.init_val_set.remove_columns(['labels']), self.init_val_set['labels'])
        for j in range(self.iter_num):
            if unfinished_tolerance >= self.early_stop_t:
                break

            if j == self.iter_num - 1:
                current_add_num = self.an_manager.get_last_iter_add_num(get_sample_num_by_class(val_set))
            else:
                current_add_num = self.an_manager.get_curr_iter_add_num(is_prev_unfinished=(unfinished_tolerance > 0))

            if unfinished_tolerance > 0:
                self.decay_threshold(ratio=4)

            curr_val_set_fd = temp_val_set_fd

            for label_index in range(len(self.label_set)):
                print(('=' * 10 + 'iter num %d class %d running' + '=' * 10) % (j, self.label_set[label_index]))
                if ignore_feature_distance:
                    val_set, curr_val_set_fd, is_finished, total_tries = \
                        self.random_sample_for_one_class(val_set,
                                                          curr_val_set_fd,
                                                          label_index,
                                                          current_add_num[self.label_set[label_index]],
                                                          j)
                else:
                    val_set, curr_val_set_fd, is_finished, total_tries = \
                        self.run_one_iter_one_class(val_set, curr_val_set_fd, label_index,
                                                    current_add_num[self.label_set[label_index]], j)

                if not is_finished:
                    # reset val set
                    val_set = temp_val_set
                    unfinished_tolerance += 1
                    print('unfinished_tolerance + 1 = %d' % (unfinished_tolerance))
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
        total_tries = 0
        curr_add_num = add_num
        remain_add_num = add_num
        try_num = 0
        label = self.label_set[label_index]
        curr_threshold = self.threshold[label]
        while remain_add_num != 0:
            curr_class_val_set = val_set[label]

            prev_class_fd = curr_val_set_fd[label]
            prev_distri_diff = self.fd_calculator. \
                cal_distribution_diff_in_class(prev_class_fd, self.whole_train_set_fd[label], label)

            if sample_method == 'RS':
                sample_indexes = np.random.randint(0, len(self.data_pool[label]), size=curr_add_num).tolist()
                samples = self.data_pool[label].select(sample_indexes)

            new_samples_fd = self.fd_calculator.cal_class_distribution(samples.remove_columns(['labels']))
            curr_class_fd = torch.cat([prev_class_fd, new_samples_fd], dim=1)
            curr_distri_diff = self.fd_calculator. \
                cal_distribution_diff_in_class(curr_class_fd, self.whole_train_set_fd[label], label)

            try_num += 1
            if prev_distri_diff - curr_distri_diff > curr_threshold:
                if sample_method == 'RS':
                    self.data_pool[label] = \
                        self.data_pool[label].select(list(set([i for i in range(len(self.data_pool[label]))]) - set(sample_indexes)))
                val_set[label] = concatenate_datasets([curr_class_val_set, samples], axis=0)
                curr_val_set_fd[label] = curr_class_fd
                print('iter num %d class %d try times %d add num %d' % (iter_index, label, try_num, curr_add_num))
                print('iter num %d class %d current distribution diff %f' % (iter_index, label, curr_distri_diff))

                remain_add_num = remain_add_num - curr_add_num
                curr_add_num = remain_add_num if remain_add_num < curr_add_num else curr_add_num
                total_tries += try_num
                try_num = 0
            else:
                if try_num > self.try_num_limits:
                    if curr_add_num == 1:
                        print('iter num %d class %d try times reach limitation' % (iter_index, label))
                        break
                    else:
                        curr_add_num = math.ceil(curr_add_num / 2)
                        curr_threshold = curr_threshold / 10
                        total_tries += try_num
                        try_num = 0

        is_finished = True if remain_add_num == 0 else False

        return val_set, curr_val_set_fd, is_finished, total_tries

    def random_sample_for_one_class(self, val_set, curr_val_set_fd, label_index, add_num, iter_index):
        label = self.label_set[label_index]
        curr_class_val_set = val_set[label]
        prev_class_fd = curr_val_set_fd[label]

        sample_indexes = np.random.randint(0, len(self.data_pool[label]), size=add_num).tolist()
        samples = self.data_pool[label].select(sample_indexes)

        new_samples_fd = self.fd_calculator.cal_class_distribution(samples.remove_columns(['labels']))
        curr_class_fd = torch.cat([prev_class_fd, new_samples_fd], dim=1)
        curr_distri_diff = self.fd_calculator. \
            cal_distribution_diff_in_class(curr_class_fd, self.whole_train_set_fd[label], label)

        self.data_pool[label] = \
            self.data_pool[label].select(
                list(set([i for i in range(len(self.data_pool[label]))]) - set(sample_indexes)))
        val_set[label] = concatenate_datasets([curr_class_val_set, samples], axis=0)
        curr_val_set_fd[label] = curr_class_fd

        print('iter num %d class %d add num %d' % (iter_index, label, add_num))
        print('iter num %d class %d current distribution diff %f' % (iter_index, label, curr_distri_diff))

        is_finished = True
        total_tries = 1

        return val_set, curr_val_set_fd, is_finished, total_tries





