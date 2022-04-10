from utils import  generate_seed_set, PathConfig
import numpy as np
import tqdm
from sklearn.metrics import pairwise_distances
import pandas as pd
from dataset_pool import get_dataset
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from datasets import Dataset
import torch


FE_SAVE_PATH = PathConfig().get_reuters_fe_path()
DATA_POOL_PATH = PathConfig().get_reuters_data_pool_path()


def get_data_features(dst, feature_extractor, device="cuda:0"):
  loader = DataLoader(dst, batch_size=64)
  feature_list = []
  for batch in loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
      outputs = feature_extractor(**batch)
      features = outputs.hidden_states[-2][:, 0, :].detach()
      feature_list.append(features.cpu())
  features = torch.cat(feature_list, dim=0)
  return features.numpy()


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

      source_ind = -1
      while source_ind == -1:
        source_inds = self.find_cloest_source_points(target_ind)
        for ind in source_inds:
          if ind not in self.already_selected:
            source_ind = ind
        if source_ind == -1:
            self.one_nn = NearestNeighbors(n_neighbors=100, algorithm='auto').fit(self.source_points)

      assert (source_ind not in self.already_selected) and (source_ind != -1)

      self.update_distances([source_ind], only_new=True, reset_dist=False)
      new_batch_source.append(source_ind)
      new_batch_target.append(target_ind)
      max_dis = max(self.curr_min_distances)
      # print('Maximum distance from cluster centers is %0.2f source_ind %d, target_ind %d' % (max_dis, source_ind, target_ind))

    return new_batch_source, new_batch_target, max_dis


class ClassStratifiedSampler(object):
  def __init__(self, feature_extractor, whole_train_set:Dataset=None, augmented_dst:Dataset=None, random_seed=0):
    self.seed = random_seed
    if whole_train_set is not None:
      self.whole_train_set = whole_train_set
      self.label_set = [0, 1]
      self.original_index2inclass_index = {}
      self.class_split_train_set = self.split_set_by_class(whole_train_set)
      self.class_split_augment_set = self.split_set_by_class(augmented_dst, setup_index=True)
    else:
      raise Exception("params missing")
    self.label_sampler_dict = {}
    for l in self.label_set:
      print(l)
      self.label_sampler_dict[l] = KCenterGreedy(get_data_features(self.class_split_augment_set[l], feature_extractor),
                                                 get_data_features(self.class_split_train_set[l], feature_extractor),
                                                 self.seed, metric='l2')

  def split_set_by_class(self, dataset: Dataset, setup_index=False):
    class_split_dset = {}
    if setup_index:
      dataset = dataset.map(lambda example, idx: {'index': idx}, with_indices=True)

    for l in self.label_set:
      if setup_index:
        class_split_dset[l] = dataset.filter(lambda example: example['labels'] == l)
        self.original_index2inclass_index[l] = pd.DataFrame({'inclass_index': [i for i in range(len(class_split_dset[l]))],
                                                           'origin_index': class_split_dset[l]['index']})
        class_split_dset[l] = class_split_dset[l].remove_columns('index')
      else:
        class_split_dset[l] = dataset.filter(lambda example: example['labels'] == l)

    if setup_index:
      dataset = dataset.remove_columns('index')
    return class_split_dset

  def holdout_sample(self, sample_num_dict=None, sample_ratio=1):
    if sample_num_dict is None:
      num_dict = {}
      for l in self.label_set:
        num_dict[l] = int(len(self.class_split_train_set[l])*sample_ratio)
      sample_num_dict = num_dict
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
  def __init__(self, whole_train_set:Dataset=None, augmented_dst:Dataset=None):
    if whole_train_set is not None:
      self.whole_train_set = whole_train_set
      self.label_set = [0, 1]
      self.original_index2inclass_index = {}
      self.class_split_train_set = self.split_set_by_class(whole_train_set)
      self.class_split_augment_set = self.split_set_by_class(augmented_dst, setup_index=True)
    else:
      raise Exception("params missing")

  def split_set_by_class(self, dataset: Dataset, setup_index=False):
    class_split_dset = {}
    if setup_index:
      dataset = dataset.map(lambda example, idx: {'index': idx}, with_indices=True)

    for l in self.label_set:
      if setup_index:
        class_split_dset[l] = dataset.filter(lambda example: example['labels'] == l)
        self.original_index2inclass_index[l] = pd.DataFrame({'inclass_index': [i for i in range(len(class_split_dset[l]))],
                                                           'origin_index': class_split_dset[l]['index']})
        class_split_dset[l] = class_split_dset[l].remove_columns('index')
      else:
        class_split_dset[l] = dataset.filter(lambda example: example['labels'] == l)

    if setup_index:
      dataset = dataset.remove_columns('index')
    return class_split_dset

  def holdout_sample(self, sample_num_dict=None, sample_ratio=1, random_seed=0):
    if sample_num_dict is None:
      num_dict = {}
      for l in self.label_set:
        num_dict[l] = int(len(self.class_split_train_set[l])*sample_ratio)
      sample_num_dict = num_dict
    val_index_list = []
    for l in self.label_set:
      indexes = self.original_index2inclass_index[l].sample(frac=1, random_state=random_seed).reset_index(drop=True)
      val_index_list.extend(indexes.iloc[0:sample_num_dict[l]]['origin_index'].values.astype(int).tolist())
    return val_index_list


def generate_init_val_for_reuters(seed_set):
  device="cuda:0"
  from transformers import AutoModelForSequenceClassification
  model_name = "distilbert-base-uncased"
  whole_train_dst, test_dst = get_dataset('wheat_corn_reuters', model_name)
  whole_train_dst.set_format("torch")
  augmented_dst = get_dataset('augmented_wheat_corn_reuters', model_name)
  print(len(augmented_dst))
  feature_extractor = AutoModelForSequenceClassification.from_pretrained(FE_SAVE_PATH,
                                                                         output_hidden_states=True)
  feature_extractor = feature_extractor.to(device)
  feature_extractor.eval()
  for s in seed_set:
    sampler = ClassStratifiedSampler(feature_extractor=feature_extractor,
                                     whole_train_set=whole_train_dst, augmented_dst=augmented_dst,
                                     random_seed=s)

    val_set_index = sampler.holdout_sample(sample_ratio=1)

    np.savetxt(DATA_POOL_PATH+'reuters_wheat_corn_valset_100p'+ str(s) +'.txt', np.asarray(val_set_index, dtype=int), fmt="%d")


def generate_init_val_for_reuters_ordered(seed_set, save_folder):
  model_name = "distilbert-base-uncased"
  whole_train_dst, test_dst = get_dataset('wheat_corn_reuters', model_name)
  whole_train_dst.set_format("torch")
  augmented_dst = get_dataset('augmented_wheat_corn_reuters', model_name)
  print(len(augmented_dst))
  sampler = ClassStratifiedSamplerByOrder(whole_train_set=whole_train_dst, augmented_dst=augmented_dst)
  for s in seed_set:
    val_set_index = sampler.holdout_sample(sample_ratio=1, random_seed=s)
    np.savetxt(save_folder + 'reuters_wheat_corn_valset_100p'+ str(s) +'.txt', np.asarray(val_set_index, dtype=int), fmt="%d")


if __name__ == '__main__':
  seed_set = generate_seed_set()
  generate_init_val_for_reuters(seed_set)
  generate_init_val_for_reuters_ordered(seed_set, save_folder=DATA_POOL_PATH+'byorder/')



