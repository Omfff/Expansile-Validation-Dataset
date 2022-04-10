import copy

import numpy as np
from sklearn.metrics import pairwise_distances
import pandas as pd
import math


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

  def __init__(self, X, seed, metric='cosine'):
    self.X = X
    self.flat_X = self.flatten_X()
    self.name = 'kcenter'
    self.features = self.flat_X
    self.metric = metric
    self.curr_min_distances = None
    self.stable_min_distances = None
    self.n_obs = self.X.shape[0]
    self.already_selected = []
    self.seed = seed

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
      x = self.features[cluster_centers]
      dist = pairwise_distances(self.features, x, metric=self.metric)

      if self.curr_min_distances is None:
        self.curr_min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.curr_min_distances = np.minimum( self.curr_min_distances, dist)

  def update_already_seleted(self, selected):
    """
        determine to add selected data point index to self.already_selected
    :param selected:
    :param is_update_distance:
    :return:
    """
    if self.curr_min_distances is None:
      self.update_distances(selected, only_new=False, reset_dist=True)
    self.stable_min_distances = self.curr_min_distances
    self.already_selected.extend(selected)

  def reset_min_distance(self):
    self.curr_min_distances = self.stable_min_distances

  def select_batch_(self, model, N, **kwargs):
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

    new_batch = []
    for _ in range(N):
      if len(self.already_selected) == 0:
        # Initialize centers with a randomly selected datapoint
        np.random.seed(self.seed)
        ind = np.random.choice(np.arange(self.n_obs))
      else:
        ind = np.argmax(self.curr_min_distances)
      # New examples should not be in already selected since those points
      # should have min_distance of zero to a cluster center.
      assert ind not in self.already_selected

      self.update_distances([ind], only_new=True, reset_dist=False)
      new_batch.append(ind)
      # self.already_selected.append(ind)
    # print('Maximum distance from cluster centers is %0.2f' % max(self.curr_min_distances))

    return new_batch


class ClassStratifiedSampler(object):
  def __init__(self, random_seed, whole_train_set=None, class_split_dst=None):
    self.seed = random_seed
    if whole_train_set is not None:
      self.whole_train_set = whole_train_set
      self.label_set = list(self.whole_train_set[1].value_counts().index)
      self.original_index2inclass_index = {}
      self.class_split_train_set = self.split_set_by_class(whole_train_set)
    elif class_split_dst is not None:
      self.class_split_train_set = copy.deepcopy(class_split_dst)
      self.label_set = class_split_dst.keys()
    else:
      raise Exception("params missing")
    self.label_sampler_dict = {}
    for l in self.label_set:
      self.label_sampler_dict[l] = KCenterGreedy(self.class_split_train_set[l], seed=self.seed)

  def split_set_by_class(self, dataset):
    dset = pd.concat([dataset[0], pd.DataFrame({'label': dataset[1].values})], axis=1)
    class_split_dset = {}
    for l in self.label_set:
      class_split_dset[l] = dset[dset['label'] == l].reset_index()
      self.original_index2inclass_index[l] = pd.concat([pd.DataFrame({'inclass_index': class_split_dset[l].index}), class_split_dset[l][['index']]], axis=1)
      class_split_dset[l] = class_split_dset[l].drop(['index', 'label'], axis=1).values
    return class_split_dset

  def holdout_sample(self, val_ratio):
    sample_num_dict = {}
    for l in self.label_set:
      sample_num_dict[l] = math.ceil(self.class_split_train_set[l].shape[0] * val_ratio)

    new_samples = {}
    val_index_list = []
    for l in self.label_set:
      new_samples[l] = self.label_sampler_dict[l].select_batch_(N=1, model=None)
      self.label_sampler_dict[l].update_already_seleted(new_samples[l])
      new_samples[l] = self.label_sampler_dict[l].select_batch_(N=sample_num_dict[l]-1, model=None)
      self.label_sampler_dict[l].update_already_seleted(new_samples[l])
      print(self.label_sampler_dict[l].already_selected)
      val_index_list.extend(self.original_index2inclass_index[l].iloc[self.label_sampler_dict[l].already_selected]['index'].values.astype(int).tolist())

    return list(set(self.whole_train_set[0].index.values.tolist()) - set(val_index_list)), val_index_list

  def test_sample(self, N_dict, already_selected_dict, model=None, update=True):
    new_samples = {}
    for l in label_set:
      new_samples[l] = self.label_sampler_dict[l].select_batch_(N=N_dict[l], model=model, already_selected=already_selected_dict[l])
      if update:
        self.label_sampler_dict[l].update_already_seleted(new_samples[l])
      else:
        self.label_sampler_dict[l].reset_min_distance()
    return new_samples

  def add_init_selected_samples(self, selected_samples_dict):
    for label in selected_samples_dict.keys():
      self.label_sampler_dict[label].update_already_seleted(selected_samples_dict[label])
      self.label_sampler_dict[label].n_obs = self.label_sampler_dict[label].n_obs - len(selected_samples_dict[label])

  def update_all_selected_samples(self, sample_index_dict):
    for l in self.label_set:
      self.label_sampler_dict[l].update_already_seleted(sample_index_dict[l])
      print("class %d updating selected samples, all selected %d"%(l, len(self.label_sampler_dict[l].already_selected)))

  def reset_all_selected_sample_to_stable_vesion(self):
      for l in self.label_set:
        self.label_sampler_dict[l].reset_min_distance()

  def sample(self, label, num, model=None):
    indexes = self.label_sampler_dict[label].select_batch_(N=num, model=model)
    return indexes


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    train_x = np.random.uniform(0,1, size=(100, 5))
    np.random.seed(seed)
    train_y = np.random.randint(0, 5, size=(100))
    label_set = [i for i in range(5)]
    dset = (pd.DataFrame(train_x, columns=['a', 'b', 'c', 'd', 'e']), pd.Series(train_y))
    temp = pd.DataFrame({'inclass_index': dset[0].index})
    sampler = ClassStratifiedSampler(whole_train_set=dset, random_seed=seed)

    train_set, val_set = sampler.holdout_sample(val_ratio=0.2)
    print(val_set)
    print(len(train_set))
    print(len(set(val_set)))
    print(len(val_set))
    # sample_num_dict = {}
    # already_selected_dict = {}
    # for l in label_set:
    #   sample_num_dict[l] = 1
    #   already_selected_dict[l] = []
    #
    # samples_round1 = sampler.sample(sample_num_dict, already_selected_dict=already_selected_dict)
    # print(samples_round1)
    # for l in label_set:
    #   sample_num_dict[l] = 3
    #
    # samples_round2 = sampler.sample(sample_num_dict, already_selected_dict=already_selected_dict)
    # print(samples_round2)
    #
    # samples_round3 = sampler.sample(sample_num_dict, already_selected_dict=already_selected_dict)
    # print(samples_round3)