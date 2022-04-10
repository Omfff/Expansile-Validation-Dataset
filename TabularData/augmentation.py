import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def CategoricalFeatureConverter_test():
    df = pd.DataFrame({'age':[0, 1, 2, 3], 'level':[1,2,3,4], 'earn':[1.9, 2.8, 3.7, 6.6], 'city':[2, 3, 4, 5], 'risk':[1.0, 2.0, 3.0, 6.0]})
    cfc = CategoricalFeatureConverter(cate_col=['age', 'level', 'city'])
    one_hot_df = cfc.convert_to_one_hot_labels(df)
    result = cfc.convert_to_category_labels(one_hot_df.values)
    if (result.values == df.values).all():
        print("success")


class CategoricalFeatureConverter(object):
    def __init__(self, cate_col):
        self.cate_col = cate_col
        self.origin_col_list = None
        self.not_cate_col = None
        self.all_col_after_one_hot_encode = None

    def convert_to_one_hot_labels(self, dst):
        self.origin_col_list = dst.columns
        self.not_cate_col = list(set(dst.columns)-set(self.cate_col))
        results = pd.get_dummies(dst, columns=self.cate_col)
        self.all_col_after_one_hot_encode = results.columns
        return results

    def convert_to_category_labels(self, dst):
        dst = pd.DataFrame(dst, columns=self.all_col_after_one_hot_encode)
        result = {}
        for c in self.cate_col:
            one_hot_col_list = []
            min_class_label = 10000000
            for nc in self.all_col_after_one_hot_encode:
                if c in nc and nc not in self.not_cate_col:
                    one_hot_col_list.append(nc)
                    curr_label = int(nc.split('_')[-1])
                    if curr_label < min_class_label:
                        min_class_label = curr_label
                    elif curr_label == min_class_label:
                        raise Exception("CategoricalFeatureConverter one hot encoding error!")

            result[c] = (np.argmax(dst[one_hot_col_list].values, axis=1)+min_class_label).reshape(-1)
        result = pd.DataFrame(result)
        result = pd.concat((result, dst[self.not_cate_col]), axis=1)
        result = result[self.origin_col_list]
        return result


class DataGenerator(object):
    def __init__(self, whole_train_set, random_seed=0, categorical_col=None, numerical_col=None):
        self.random_seed = random_seed
        self.whole_train_set = whole_train_set
        self.data_pool_expand_ratio = 30
        self.data_pool = {}
        self.label_set = list(self.whole_train_set[1].value_counts().index)
        self.categorical_col = categorical_col

    def split_set_by_class(self, dataset):
        dset = pd.concat([dataset[0], pd.DataFrame({'label':dataset[1].values})], axis=1)
        class_split_dset = {}
        for l in self.label_set:
            class_split_dset[l] = dset[dset['label']==l].reset_index(drop=True)
        return class_split_dset

    def merge_set(self, dst):
        columns = self.whole_train_set[0].columns.values.tolist()
        columns.append('label')
        merged_set = pd.DataFrame(columns=columns)
        for label in self.label_set:
            merged_set = merged_set.append(pd.DataFrame(dst[label], columns=columns)
                                           , ignore_index=True)
        merged_set = merged_set.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        return merged_set.drop(['label'], axis=1), merged_set['label']

    def generate_data_to_pool(self, method):
        """ Generate auxiliary dataset

        :param method: RS|SMOTE|mixup
        :return: auxiliary dataset
        """
        np.random.seed(self.random_seed)
        if method == 'SMOTE':
            cf_converter = CategoricalFeatureConverter(self.categorical_col)
            dst = self.split_set_by_class((cf_converter.convert_to_one_hot_labels(self.whole_train_set[0]), self.whole_train_set[1]))
        else:
            dst = self.split_set_by_class(self.whole_train_set)

        for label, samples in dst.items():
            expand_size = self.data_pool_expand_ratio * len(samples)
            if method == 'RS':
                self.data_pool[label] = samples.sample(frac=self.data_pool_expand_ratio,
                                                       replace=True, random_state=self.random_seed).values
            elif method == 'SMOTE':
                samples = samples.drop(['label'], axis=1).values
                nns = NearestNeighbors(n_neighbors=5).fit(samples).kneighbors(samples, return_distance=False)[:, 1:]
                samples_indices = np.random.randint(low=0, high=np.shape(samples)[0], size=expand_size)
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


if __name__ == '__main__':
    CategoricalFeatureConverter_test()



