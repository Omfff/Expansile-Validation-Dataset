import copy
import numpy as np
from sklearn.metrics import pairwise_distances
import pandas as pd
import math
from sklearn.neighbors import NearestNeighbors
from sample import SamplingMethod


class KCenterGreedy(SamplingMethod):
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
    self.one_nn = NearestNeighbors(n_neighbors=20, algorithm='auto').fit(source_points)

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
      #   self.one_nn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(self.source_points)

      assert (source_ind not in self.already_selected) and (source_ind != -1)

      self.update_distances([source_ind], only_new=True, reset_dist=False)
      new_batch_source.append(source_ind)
      new_batch_target.append(target_ind)
      max_dis = max(self.curr_min_distances)
      # print('Maximum distance from cluster centers is %0.2f source_ind %d, target_ind %d' % (max_dis, source_ind, target_ind))

    return new_batch_source, new_batch_target, max_dis


class CoresetSampler(object):
  def __init__(self, random_seed, whole_train_set=None, data_pool=None):
    self.seed = random_seed
    if whole_train_set is not None:
      self.whole_train_set = whole_train_set
      self.label_set = list(self.whole_train_set[1].value_counts().index)
      self.original_index2inclass_index = {}
      # the order can not change!
      self.class_split_train_set = self.split_set_by_class(whole_train_set)
      self.class_split_data_pool = self.split_set_by_class(data_pool)
    else:
      raise Exception("params missing")
    self.label_sampler_dict = {}
    for l in self.label_set:
      self.label_sampler_dict[l] = KCenterGreedy(self.class_split_data_pool[l], self.class_split_train_set[l],
                                                 seed=self.seed)

  def split_set_by_class(self, dataset):
    dset = pd.concat([dataset[0], pd.DataFrame({'label': dataset[1].values})], axis=1)
    class_split_dset = {}
    for l in self.label_set:
      class_split_dset[l] = dset[dset['label'] == l].reset_index()
      self.original_index2inclass_index[l] = pd.concat([pd.DataFrame({'inclass_index': class_split_dset[l].index}),
                                                        class_split_dset[l][['index']]], axis=1)
      class_split_dset[l] = class_split_dset[l].drop(['index', 'label'], axis=1).values
    return class_split_dset

  def coreset_sample(self, val_ratio):
    sample_num_dict = {}
    for l in self.label_set:
      sample_num_dict[l] = math.ceil(self.class_split_train_set[l].shape[0] * val_ratio)

    new_samples = {}
    val_index_list = []
    for l in self.label_set:
      final_max_dis = -1
      for num in range(sample_num_dict[l]):
        new_samples[l], train_index, dis = self.label_sampler_dict[l].select_batch_(N=1)
        final_max_dis = dis[0]
        self.label_sampler_dict[l].update_already_seleted(new_samples[l], train_index)
      print("final max distance for class %d is %0.2f" % (l, final_max_dis))
      val_index_list.extend(self.original_index2inclass_index[l].iloc[self.label_sampler_dict[l].already_selected]['index'].values.astype(int).tolist())

    return val_index_list


class RandomCoresetSampler(object):
  def __init__(self, random_seed, whole_train_set=None, data_pool=None):
    self.seed = random_seed
    if whole_train_set is not None:
      self.whole_train_set = whole_train_set
      self.label_set = list(self.whole_train_set[1].value_counts().index)
      self.original_index2inclass_index = {}
      # the order can not change!
      self.class_split_train_set = self.split_set_by_class(whole_train_set)
      self.class_split_data_pool = self.split_set_by_class(data_pool)
    else:
      raise Exception("params missing")

  def split_set_by_class(self, dataset):
    dset = pd.concat([dataset[0], pd.DataFrame({'label': dataset[1].values})], axis=1)
    class_split_dset = {}
    for l in self.label_set:
      class_split_dset[l] = dset[dset['label'] == l].reset_index()
      self.original_index2inclass_index[l] = pd.concat([pd.DataFrame({'inclass_index': class_split_dset[l].index}),
                                                        class_split_dset[l][['index']]], axis=1)
      class_split_dset[l] = class_split_dset[l].drop(['index', 'label'], axis=1).values
    return class_split_dset

  def coreset_sample(self, val_ratio):
    np.random.seed(self.seed)
    sample_num_dict = {}
    for l in self.label_set:
      sample_num_dict[l] = math.ceil(self.class_split_train_set[l].shape[0] * val_ratio)

    val_index_list = []
    for l in self.label_set:
      inclass_indexes = np.random.randint(0, len(self.original_index2inclass_index[l]), size=sample_num_dict[l])
      val_index_list.extend(self.original_index2inclass_index[l].iloc[inclass_indexes]['index'].values.astype(int).tolist())

    return val_index_list
