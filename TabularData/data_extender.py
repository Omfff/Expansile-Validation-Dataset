import math
import copy
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from feature_distribution import FeatureDistribution
from augmentation import CategoricalFeatureConverter


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
    def __init__(self, whole_train_set, init_train_set, init_val_set, fd:FeatureDistribution=None, iter_num=None,
                 add_ratio_per_iter=None, diff_threshold_ratio=0.1, early_stop_threshold=3, try_num_limits=300,
                 add_num_decay_rate=0.5, add_num_decay_method=None, add_num_decay_stage=None,
                 random_seed = 0,
                 categorical_col=None, numerical_col=None):
        self.random_seed = random_seed
        self.whole_train_set = whole_train_set
        self.train_set = init_train_set
        self.init_val_set = init_val_set
        self.iter_num = iter_num
        self.add_ratio = add_ratio_per_iter
        # self.threshold = diff_threshold
        self.data_pool_expand_ratio = 30
        self.data_pool = {}

        self.fd_calculator = fd
        self.whole_train_set_fd = self.fd_calculator.cal_distribution(self.whole_train_set[0], self.whole_train_set[1])

        self.init_val_fd = self.fd_calculator.cal_distribution(self.init_val_set[0], self.init_val_set[1])
        self.init_fd_diff = self.fd_calculator.cal_distribution_diff(self.whole_train_set_fd, self.init_val_fd)

        self.label_set = list(self.init_val_set[1].value_counts().index)
        self.class_split_dst = self.split_set_by_class(self.init_val_set)
        self.add_num_per_iter = self.cal_add_num_per_iter()

        self.threshold = self.__init_threshold(diff_threshold_ratio)
        # {l:self.init_fd_diff[l]*diff_threshold_ratio for l in self.label_set}

        self.early_stop_t = early_stop_threshold
        self.try_num_limits = try_num_limits

        self.an_manager = SampleAddNumManager(get_sample_num_by_class(self.class_split_dst), self.add_num_per_iter,
                                              decay_rate=add_num_decay_rate, decay_method=add_num_decay_method, stage=add_num_decay_stage)

        self.categorical_col = categorical_col

    def __init_threshold(self, diff_threshold_ratio):
        total_diff = np.sum(list(self.init_fd_diff.values()))
        return {l: self.init_fd_diff[l] * (diff_threshold_ratio*(self.init_fd_diff[l]/total_diff)) for l in self.label_set}

    def split_set_by_class(self, dataset):
        dset = pd.concat([dataset[0], pd.DataFrame({'label':dataset[1].values})], axis=1)
        class_split_dset = {}
        for l in self.label_set:
            class_split_dset[l] = dset[dset['label']==l].reset_index(drop=True)
        return class_split_dset

    def cal_add_num_per_iter(self):
        result = {}
        for label, samples in self.class_split_dst.items():
           result[label] = int(len(samples) * self.add_ratio)
        return result

    def merge_set(self, dst):
        merged_set = pd.DataFrame(columns=dst[self.label_set[0]].columns)
        for label in self.label_set:
            merged_set = merged_set.append(dst[label], ignore_index=True)
        merged_set = merged_set.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        return merged_set.drop(['label'], axis=1), merged_set['label']

    def generate_data_to_pool(self, method):
        """

        :param dst:
        :param method: RS|SMOTE|mixup
        :return:
        """
        # class_sample_index_dict = {}
        # for i in range(len(self.init_val_set[1])):
        #     label = self.init_val_set[1].loc[i]
        #     if label not in class_sample_index_dict.keys():
        #         class_sample_index_dict[label] = []
        #     class_sample_index_dict[label].append(i)
        # for label, indexes in class_sample_index_dict:
        #     self.add_num_per_iter[label] = int(len(indexes) * self.add_ratio)
        np.random.seed(self.random_seed)
        if method == 'SMOTE':
            cf_converter = CategoricalFeatureConverter(self.categorical_col)
            dst = self.split_set_by_class((cf_converter.convert_to_one_hot_labels(self.train_set[0]), self.train_set[1]))
        else:
            dst = self.split_set_by_class(self.train_set)

        for label, samples in dst.items():
            expand_size = self.data_pool_expand_ratio * len(samples)
            if method == 'RS':
                self.data_pool[label] = samples.sample(frac=self.data_pool_expand_ratio,
                                                       replace=True, random_state=self.random_seed).values
            elif method == 'SMOTE':
                samples = samples.drop(['label'], axis=1).values
                # 获取每一个少数类样本点周围最近的n_neighbors-1个点的位置矩阵
                nns = NearestNeighbors(n_neighbors=5).fit(samples).kneighbors(samples, return_distance=False)[:, 1:]
                # 随机产生diff个随机数作为之后产生新样本的选取的样本下标值
                samples_indices = np.random.randint(low=0, high=np.shape(samples)[0], size=expand_size)
                # 随机产生diff个随机数作为之后产生新样本的间距值
                steps = np.random.uniform(size=expand_size)
                cols = np.mod(samples_indices, nns.shape[1])
                expand_samples = np.zeros((expand_size, samples.shape[1]))
                for i, (col, step) in enumerate(zip(cols, steps)):
                    row = samples_indices[i]
                    expand_samples[i] = samples[row] - step * (
                                samples[row] - samples[nns[row, col]])
                pool_samples_in_label = np.concatenate((expand_samples, label*np.ones((expand_size,1), dtype=int)), axis=1)
                self.data_pool[label] = np.concatenate((
                    cf_converter.convert_to_category_labels(pool_samples_in_label[:,0:-1]).values,
                    pool_samples_in_label[:,-1].reshape(-1, 1)), axis=1)
            elif method == 'MIXUP':
                indexes = np.random.randint(low=0, high=len(samples), size=(expand_size, 2))
                samples = samples.drop(['label'], axis=1).values
                # alpha = np.repeat(np.expand_dims(np.random.beta(0.5, 0.5, size=expand_size), axis=1), samples.shape[1], axis=1)
                alpha = np.expand_dims(np.random.beta(0.5, 0.5, size=expand_size), axis=1)
                new_samples = alpha*samples[indexes[:, 0]] + (1-alpha)*samples[indexes[:, 1]]
                self.data_pool[label] = np.concatenate((new_samples, label*np.ones((expand_size,1), dtype=int)), axis=1)
        return self.data_pool

    def check_label_distribution_change(self, dst1, dst2):
        change_rate = {}
        for l in dst1.keys():
            change_rate[l] = len(dst2[l])/len(dst1[l])
        print("label_distribution_change", change_rate)

    def decay_threshold(self, ratio=2):
        for l in self.label_set:
            self.threshold[l] = self.threshold[l]/ratio

    def run(self, sample_method):
        """

        :param sample_method: RS
        :return:
        """
        unfinished_tolerance = 0
        print("init_fd_diff", self.init_fd_diff)
        print("init_threshold", self.threshold)

        val_set = copy.deepcopy(self.class_split_dst)
        temp_val_set = copy.deepcopy(self.class_split_dst)
        temp_val_set_fc = self.fd_calculator.cal_distribution(self.init_val_set[0], self.init_val_set[1], normalize=False)
        for j in range(self.iter_num):
            if unfinished_tolerance >= self.early_stop_t:
                break

            if j == self.iter_num-1:
                current_add_num = self.an_manager.get_last_iter_add_num(get_sample_num_by_class(val_set))
            else:
                current_add_num = self.an_manager.get_curr_iter_add_num(is_prev_unfinished=(unfinished_tolerance > 0))

            if unfinished_tolerance > 0:
                self.decay_threshold(ratio=10)

            curr_val_set_fc = temp_val_set_fc

            for label_index in range(len(self.label_set)):
                print(('=' * 10 + 'iter num %d class %d running' + '=' * 10) % (j, self.label_set[label_index]))
                val_set, curr_val_set_fc , is_finished, total_tries = \
                    self.run_one_iter_one_class(val_set, curr_val_set_fc, label_index,
                                       current_add_num[self.label_set[label_index]], j)

                if not is_finished:
                    # reset val set
                    val_set = temp_val_set
                    unfinished_tolerance += 1
                    print('unfinished_tolerance + 1 = %d'% (unfinished_tolerance))
                    break

                if label_index == len(self.label_set) - 1 and is_finished:
                    temp_val_set = copy.deepcopy(val_set)
                    temp_val_set_fc = copy.deepcopy(curr_val_set_fc)
                    unfinished_tolerance = 0

        self.check_label_distribution_change(self.class_split_dst, val_set)
        val_set = self.merge_set(val_set)

        return self.train_set, val_set

    def run_one_iter_one_class(self, val_set, curr_val_set_fc, label_index, add_num, iter_index):
        total_tries = 0
        curr_add_num = add_num
        remain_add_num = add_num
        try_num = 0
        label = self.label_set[label_index]
        curr_threshold = self.threshold[label]
        curr_class_val_set = val_set[label]
        while remain_add_num != 0:
            label = self.label_set[label_index]
            prev_class_fc = curr_val_set_fc[label]
            prev_distri_diff = self.fd_calculator. \
                cal_distribution_diff_in_class(self.whole_train_set_fd[label],
                                               self.fd_calculator.normalize_feature_frequence_to_distribution(
                                                   prev_class_fc))

            sample_indexes = np.random.randint(0, len(self.data_pool[label]), size=curr_add_num)
            samples = self.data_pool[label][sample_indexes]

            new_samples_fc = self.fd_calculator.cal_class_distribution(
                pd.DataFrame(samples, columns=curr_class_val_set.columns), label, normalize=False)
            curr_class_fc = self.fd_calculator.add_in_class_feature_distribution(new_samples_fc, prev_class_fc)
            curr_distri_diff = self.fd_calculator. \
                cal_distribution_diff_in_class(self.whole_train_set_fd[label],
                                               self.fd_calculator.normalize_feature_frequence_to_distribution(
                                                   curr_class_fc))

            try_num += 1
            if prev_distri_diff - curr_distri_diff > curr_threshold:
                self.data_pool[label] = np.delete(self.data_pool[label], sample_indexes, axis=0)
                val_set[label] = curr_class_val_set.append(pd.DataFrame(samples, columns=curr_class_val_set.columns),
                                                           ignore_index=True)
                curr_val_set_fc[label] = curr_class_fc
                print('iter num %d class %d try times %d' % (iter_index, label, try_num))
                print('iter num %d class %d current distribution diff %f' % (iter_index, label, curr_distri_diff))
                remain_add_num = remain_add_num - curr_add_num
                curr_add_num = remain_add_num if remain_add_num < curr_add_num else curr_add_num
                total_tries += try_num
                try_num = 0
            else:
                if try_num > self.try_num_limits:
                    if curr_add_num == 1 and not (add_num <=2 and total_tries < self.try_num_limits*2):
                        print('iter num %d class %d try times reach limitation' % (iter_index, label))
                        break
                    else:
                        curr_add_num = math.ceil(curr_add_num / 2)
                        curr_threshold = curr_threshold / 10
                        total_tries += try_num
                        try_num = 0

        is_finished = True if remain_add_num == 0 else False
        return val_set, curr_val_set_fc, is_finished, total_tries

    def run_random_select_without_limitation(self):
        print(self.init_fd_diff)
        val_set = copy.deepcopy(self.class_split_dst)
        curr_val_set_fc = self.fd_calculator.cal_distribution(self.init_val_set[0], self.init_val_set[1], normalize=False)
        for j in range(self.iter_num):
            if j == self.iter_num-1:
                current_add_num = self.an_manager.get_last_iter_add_num(get_sample_num_by_class(val_set))
            else:
                current_add_num = self.an_manager.get_curr_iter_add_num(is_prev_unfinished=False)

            label_index = 0

            while label_index < len(self.label_set) and current_add_num[self.label_set[label_index]] > 0:
                label = self.label_set[label_index]

                curr_class_val_set = val_set[label]
                prev_class_fc = curr_val_set_fc[label]
                prev_distri_diff = self.fd_calculator.\
                    cal_distribution_diff_in_class(self.whole_train_set_fd[label],
                                        self.fd_calculator.normalize_feature_frequence_to_distribution(prev_class_fc))

                sample_indexes = np.random.randint(0, len(self.data_pool[label]), size=current_add_num[label])
                samples = self.data_pool[label][sample_indexes]

                new_samples_fc = self.fd_calculator.cal_class_distribution(
                                        pd.DataFrame(samples, columns=curr_class_val_set.columns), label, normalize=False)
                curr_class_fc = self.fd_calculator.add_in_class_feature_distribution(new_samples_fc, prev_class_fc)
                curr_distri_diff = self.fd_calculator. \
                    cal_distribution_diff_in_class(self.whole_train_set_fd[label],
                                                   self.fd_calculator.normalize_feature_frequence_to_distribution(
                                                       curr_class_fc))

                self.data_pool[label] = np.delete(self.data_pool[label], sample_indexes, axis=0)
                val_set[label] = curr_class_val_set.append(pd.DataFrame(samples, columns=curr_class_val_set.columns),
                                                                        ignore_index=True)
                curr_val_set_fc[label] = curr_class_fc
                print('iter num %d class %d current distribution diff %f'%(j, label, curr_distri_diff))
                label_index += 1

        self.check_label_distribution_change(self.class_split_dst, val_set)
        val_set = self.merge_set(val_set)

        return self.train_set, val_set
