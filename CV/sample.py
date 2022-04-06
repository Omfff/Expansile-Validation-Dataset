import copy

import numpy as np
import tqdm
from sklearn.metrics import pairwise_distances
import pandas as pd
import math
from imbalanced_dataset import get_dataset, DatasetWrapper, update_transform
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader


SAMPLE_INDEX_SAVE_PATH = 'data_indexes'


def get_data_features(dst:DatasetWrapper):
  """ Extract features from dst(contains N samples) and return feature matrix (N*M, M is the feature dimension)

  :param dst:
  :return:
  """
  from feature_extractor import load_feature_extractor
  import torch
  device = "cuda:0"
  feature_extractor = load_feature_extractor('feature_extractor/model.th', device)
  feature_extractor.eval()
  loader = DataLoader(dst, batch_size=64, shuffle=False, num_workers=8, pin_memory=False)
  features_list = []
  with torch.no_grad():
    for datas, _ in loader:
      datas = datas.to(device)
      _, features = feature_extractor(datas)
      features_list.append(features.detach().cpu())
  features_list = torch.cat(features_list, dim=0)
  return features_list.numpy()


class SamplingMethod(object):
  def __init__(self, X, y, seed, **kwargs):
    self.X = X
    self.y = y
    self.seed = seed

  def flatten_X(self):
    shape = self.X.shape
    flat_X = self.X
    if len(shape) > 2:
      flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
    return flat_X

  def select_batch_(self):
    return

  def select_batch(self, **kwargs):
    return self.select_batch_(**kwargs)

  def to_dict(self):
    return None


class KCenterGreedy(SamplingMethod):
  """ Origin code from https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py """

  def __init__(self, source_points, target_points, seed, metric='l2'):#'cosine
    self.source_points = source_points
    self.name = 'kcenter'
    self.target_points = target_points
    self.metric = metric
    self.curr_min_distances = None
    self.stable_min_distances = None
    self.n_obs = self.target_points.shape[0]
    self.already_selected = []
    self.seed = seed
    self.one_nn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(source_points)

  def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
    """Update min distances given cluster centers.
    Args:
      cluster_centers: indices of cluster centers
      only_new: only calculate distance for newly selected points and update
        min_distances.
      rest_dist: whether to reset min_distances.
    """

    if reset_dist:
      self.curr_min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
    if cluster_centers:
      # Update min_distances for all examples given new cluster center.
      x = self.source_points[cluster_centers]
      dist = pairwise_distances(self.target_points, x, metric=self.metric)

      if self.curr_min_distances is None:
        self.curr_min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.curr_min_distances = np.minimum(self.curr_min_distances, dist)

  def update_already_seleted(self, source_selected, target_selected):
    """
        determine to add selected data point index to self.already_selected
    :param selected:
    :param is_update_distance:
    :return:
    """
    if self.curr_min_distances is None:
      self.update_distances(source_selected, only_new=False, reset_dist=True)
    self.curr_min_distances[target_selected] = 0
    self.stable_min_distances = self.curr_min_distances
    self.already_selected.extend(source_selected)

  def reset_min_distance(self):
    self.curr_min_distances = self.stable_min_distances

  def find_cloest_source_points(self, target_ind):
    return self.one_nn.kneighbors(self.target_points[target_ind].reshape(1, -1), return_distance=False)[0, :]

  def select_batch_(self, N, **kwargs):
    """
    Diversity promoting active learning method that greedily forms a batch
    to minimize the maximum distance to a cluster center among all unlabeled
    datapoints.
    Args:
      model: model with scikit-like API with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size
    Returns:
      indices of points selected to minimize distance to cluster centers
    """

    new_batch_target = []
    new_batch_source = []
    for _ in range(N):
      if len(self.already_selected) == 0:
        # Initialize centers with a randomly selected datapoint
        np.random.seed(self.seed)
        target_ind = np.random.choice(np.arange(self.n_obs))
      else:
        target_ind = np.argmax(self.curr_min_distances)
      # New examples should not be in already selected since those points
      # should have min_distance of zero to a cluster center.

      source_inds = self.find_cloest_source_points(target_ind)
      source_ind = -1
      for ind in source_inds:
        if ind not in self.already_selected:
          source_ind = ind
      # if source_ind == -1:
      #   self.one_nn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(source_points)

      assert (source_ind not in self.already_selected) and (source_ind != -1)

      self.update_distances([source_ind], only_new=True, reset_dist=False)
      new_batch_source.append(source_ind)
      new_batch_target.append(target_ind)
      max_dis = max(self.curr_min_distances)
      # print('Maximum distance from cluster centers is %0.2f source_ind %d, target_ind %d' % (max_dis, source_ind, target_ind))

    return new_batch_source, new_batch_target, max_dis


class ClassStratifiedSampler(object):
  def __init__(self, random_seed, whole_train_set:DatasetWrapper=None, augmented_dst:DatasetWrapper=None,
               class_split_dst=None):
    """ Select initial validation set from auxiliary dataset by coreset operation

    :param random_seed:
    :param whole_train_set: source set
    :param augmented_dst: auxiliary dataset
    :param class_split_dst:
    """
    self.seed = random_seed
    if whole_train_set is not None:
      self.whole_train_set = whole_train_set
      self.label_set = whole_train_set.get_label_set()
      self.original_index2inclass_index = {}
      self.class_split_train_set = self.split_set_by_class(whole_train_set)
      self.class_split_augment_set = self.split_set_by_class(augmented_dst)
    elif class_split_dst is not None:
      self.class_split_train_set = copy.deepcopy(class_split_dst)
      self.label_set = class_split_dst.keys()
    else:
      raise Exception("params missing")
    self.label_sampler_dict = {}
    # for each class, create a kcenter greedy sampler
    for l in self.label_set:
      self.label_sampler_dict[l] = KCenterGreedy(get_data_features(self.class_split_augment_set[l]),
                                                 get_data_features(self.class_split_train_set[l]),
                                                 self.seed, metric='l2')

  def split_set_by_class(self, dataset:DatasetWrapper):
    dset = dataset
    class_split_dset = {}
    for l in self.label_set:
      class_split_dset[l] = dset.get_dataset_by_class(l)
      self.original_index2inclass_index[l] = pd.DataFrame({'inclass_index': class_split_dset[l].indexset,
                                                           'origin_index':dataset.class_split_indexes[l]})
      class_split_dset[l] = class_split_dset[l]
    return class_split_dset

  def holdout_sample(self, sample_num_dict):
    """
    :param sample_num_dict: {category: number of selectd samples, ...}
    :return: a list contains selected sample index
    """
    new_samples = {}
    val_index_list = []
    for l in self.label_set:
      final_max_dis = -1
      for num in tqdm.tqdm(range(sample_num_dict[l])):
        new_samples[l], train_index, dis = self.label_sampler_dict[l].select_batch_(N=1)
        final_max_dis = dis[0]
        self.label_sampler_dict[l].update_already_seleted(new_samples[l], train_index)
      print("final max distance for class %d is %0.2f" % (l, final_max_dis))
      val_index_list.extend(self.original_index2inclass_index[l]
                            .iloc[self.label_sampler_dict[l].already_selected]
                            ['origin_index'].values.astype(int).tolist())
    return val_index_list


class ClassStratifiedSamplerByOrder(object):
  def __init__(self, whole_train_set:DatasetWrapper=None, augmented_dst:DatasetWrapper=None,
               class_split_dst=None):
    if whole_train_set is not None:
      self.whole_train_set = whole_train_set
      self.label_set = whole_train_set.get_label_set()
      self.original_index2inclass_index = {}
      self.class_split_train_set = self.split_set_by_class(whole_train_set)
      self.class_split_augment_set = self.split_set_by_class(augmented_dst)
    elif class_split_dst is not None:
      self.class_split_train_set = copy.deepcopy(class_split_dst)
      self.label_set = class_split_dst.keys()
    else:
      raise Exception("params missing")

  def split_set_by_class(self, dataset:DatasetWrapper):
    dset = dataset
    class_split_dset = {}
    for l in self.label_set:
      class_split_dset[l] = dset.get_dataset_by_class(l)
      self.original_index2inclass_index[l] = pd.DataFrame({'inclass_index': class_split_dset[l].indexset,
                                                           'origin_index':dataset.class_split_indexes[l]})
      class_split_dset[l] = class_split_dset[l]
    return class_split_dset

  def holdout_sample(self, sample_num_dict, random_seed):
    val_index_list = []
    for l in self.label_set:
      indexes = self.original_index2inclass_index[l].sample(frac=1, random_state=random_seed).reset_index(drop=True)
      val_index_list.extend(indexes.iloc[0:sample_num_dict[l]]['origin_index'].values.astype(int).tolist())
    return val_index_list


def generate_init_val_for_cifar10(seed_set):
  """ Select initial validation set from auxiliary dataset by coreset operation and save indexes

  :param seed_set: random seed set
  """
  train_dst = get_dataset('im_cifar10', split='train')
  update_transform(train_dst, t_type='test')

  augment_dst = get_dataset('aug_cifar10')
  update_transform(augment_dst, t_type='test')

  for s in seed_set:
    sampler = ClassStratifiedSampler(whole_train_set=train_dst, augmented_dst=augment_dst, random_seed=s)

    val_set_index = sampler.holdout_sample(train_dst.dataset.num_per_cls_dict)

    np.savetxt(SAMPLE_INDEX_SAVE_PATH+'/cifar10_1p_valset_100p'+ str(s) +'.txt', np.asarray(val_set_index, dtype=int), fmt="%d")


def generate_init_val_for_cifar10_byorder(seed_set, save_folder):
  """ Select initial validation set from auxiliary dataset by random order

  :param seed_set: random seed for shuffle
  """
  train_dst = get_dataset('im_cifar10', split='train')
  update_transform(train_dst, t_type='test')

  augment_dst = get_dataset('aug_cifar10')
  update_transform(augment_dst, t_type='test')
  sampler = ClassStratifiedSamplerByOrder(whole_train_set=train_dst, augmented_dst=augment_dst)
  for s in seed_set:
    val_set_index = sampler.holdout_sample(train_dst.dataset.num_per_cls_dict, s)
    np.savetxt(save_folder+'cifar10_1p_valset_100p'+ str(s) +'.txt', np.asarray(val_set_index, dtype=int), fmt="%d")


def load_dst(s):
  dst_indexes = np.loadtxt(SAMPLE_INDEX_SAVE_PATH+'/cifar10_1p_valset_100p'+str(s)+'.txt')
  dst_indexes = dst_indexes.astype(dtype=int)
  return dst_indexes
  augment_dst = get_dataset('aug_cifar10')
  dst = DatasetWrapper(augment_dst.get_dataset_by_indexes(dst_indexes))
  del augment_dst
  for l in dst.class_split_indexes.keys():
    print(len(dst.class_split_indexes[l]))


def check_val_set_diversity(seed_set, index_folder):
  """
    Check the overlap of the validation sets under different random seeds
  """
  valset_set = []
  augment_dst = get_dataset('aug_cifar10')
  for s in seed_set:
    dst_indexes = np.loadtxt(index_folder+'cifar10_1p_valset_100p'+str(s)+'.txt')
    dst_indexes = dst_indexes.astype(dtype=int)
    dst = DatasetWrapper(augment_dst.get_dataset_by_indexes(dst_indexes))
    class_indexes = dst.class_split_indexes
    results = {}
    for key, ids in class_indexes.items():
      results[key] = dst_indexes[ids]
    valset_set.append(results)

  total_count_dict = {}
  one_set_count_dict = {}
  overlap_ratio_dict = {}
  for key in valset_set[0].keys():
    temp = set()
    for valset in valset_set:
      temp = temp | set(valset[key])
    total_count_dict[key] = len(temp)
    one_set_count_dict[key] = len(valset_set[0][key])
    overlap_ratio_dict[key] = one_set_count_dict[key]/total_count_dict[key]

  print(one_set_count_dict)
  print(total_count_dict)
  print(overlap_ratio_dict)


if __name__ == '__main__':
  np.random.seed(0)  # 0
  seed_set = np.random.randint(0, 10000, size=10).tolist()
  generate_init_val_for_cifar10(seed_set)

  # generate_init_val_for_cifar10_byorder(seed_set, save_folder=SAMPLE_INDEX_SAVE_PATH+'/byorder/')
  #
  # check_val_set_diversity(seed_set, index_folder=SAMPLE_INDEX_SAVE_PATH+'/byorder/')
