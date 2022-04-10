"""
original code from https://github.com/henrymoss/COLING2018/blob/master/Interactive/REUTERS/Figure_5_and_Table_3.ipynb
 and https://github.com/henrymoss/COLING2018/blob/master/non-interactive/Reuters/REUTERSSVM_tuner.py
"""
from nltk.corpus import reuters
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")
from dataset_pool import decode_to_sentence, get_dataset, post_process
from feature_distribution import FeatureDistribution, FeatureExtractorType
from data_extender import DataExtender
from utils import split_train_val
from copy import deepcopy
from datasets import concatenate_datasets
import matplotlib
from scipy import stats
import pandas as pd
import os
import pickle
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
from tqdm import tqdm
from utils import PathConfig

PC = PathConfig()
DATA_POOL_PATH = PC.get_reuters_data_pool_path()
FE_SAVE_PATH = PC.get_reuters_fe_path()
GS_SAVE_PATH = PC.get_reuters_gs_path()
from mpl_toolkits.mplot3d import Axes3D
################################
#script to perform CV on RF for Movie and select max_features 500 for 500 partition choices
#output: results (1000 by #parameter options pickle)
##############################

#set seeds for reproducibility
random.seed(1234)
rs=1234



def save_reuters_dst_to_disk(save_path):
	#select only articles about wheat and/or corn
	wheat_docs = reuters.fileids("wheat")
	corn_docs = reuters.fileids("corn")
	wheat_train = list(filter(lambda doc: doc.startswith("train"),wheat_docs))
	wheat_test = list(filter(lambda doc: doc.startswith("test"),wheat_docs))
	corn_train = list(filter(lambda doc: doc.startswith("train"),corn_docs))
	corn_test = list(filter(lambda doc: doc.startswith("test"),corn_docs))
	training_index = wheat_train +wheat_test+ corn_train+corn_test
	#prepare data for wheat vs not wheat case
	text=[]
	clas = []
	classname = ["pos", "neg"]
	for i in training_index:
			text.append(reuters.raw(i))
			#check categorisation to make response
			if "wheat" in reuters.categories(i):
				clas.append(1)
			else:
				clas.append(0)
	#store in dataframe
	data = pd.DataFrame(clas, columns=['label'])
	data["text"] = text
	data = shuffle(data)
	print("We have "+str(len(text))+" classified examples")
	data.to_csv(save_path, index=False)
	return data


def get_data_for_grid_search(whole_dst, num_classes, k, seed, model_name, fe_weight_path, is_aug=False, device="cuda:0",
							 data_extender_args=None):
	num_k = k

	train_val_index_list = split_train_val([i for i in range(len(whole_dst))], whole_dst['labels'],
										   seed=seed, k=num_k)

	if not is_aug:
		final_dst = decode_to_sentence(whole_dst, model_name)
		final_dst = final_dst.remove_columns(['input_ids', 'attention_mask']).rename_column('labels', 'label')
		return final_dst.to_pandas(), train_val_index_list
	else:
		feature_distributor = FeatureDistribution(labels=[i for i in range(num_classes)], device=device,
												  weight_path=fe_weight_path,
												  dis_type='NDB',
												  feature_extractor_type=FeatureExtractorType.FineTune)
		np.random.seed(seed)
		data_extender_seeds = np.random.randint(0, 10000, size=num_k).tolist()
		print(data_extender_seeds)
		aug_train_val_index_list = []
		final_dst = deepcopy(whole_dst)
		for i in range(num_k):
			print(len(whole_dst))
			train_index, val_index = train_val_index_list[i]
			train_dst = whole_dst.select(train_index)
			val_dst = whole_dst.select(val_index)

			train_val_feature_distribution_diff = feature_distributor.cal_distribution_diff_for_two_set(
				val_dst.remove_columns(['labels']), val_dst['labels'], whole_dst.remove_columns(['labels']),
				whole_dst['labels'])
			print('train_val_feature_distribution_diff', train_val_feature_distribution_diff)
			data_extender = DataExtender(whole_dst, train_dst, val_dst, fd=feature_distributor, **data_extender_args,
										 random_seed=data_extender_seeds[i])
			data_extender.generate_data_to_pool(DATA_POOL_PATH+'pool.csv', post_process, model_name)
			train_dst, val_dst = data_extender.run()

			aug_train_val_index_list.append((train_index, [j+len(final_dst) for j in range(len(val_dst))]))
			final_dst = concatenate_datasets([final_dst, val_dst], axis=0)

		final_dst = decode_to_sentence(final_dst, model_name)
		final_dst = final_dst.remove_columns(['input_ids', 'attention_mask']).rename_column('labels', 'label')
		return final_dst.to_pandas(), aug_train_val_index_list


#create a classifier object with the classifier and parameter candidates
	# this fixes partition aross evaluations
def crossvalidate(whole_dst, parameter_candidates, num_classes, num_folds, random_seed, data_extender_args, is_aug):
	data, iterable_indexes = get_data_for_grid_search(whole_dst, num_classes=num_classes, k=num_folds, seed=random_seed,
													  model_name="distilbert-base-uncased",
													  fe_weight_path=FE_SAVE_PATH,
													  is_aug=is_aug, device="cuda:0",
													  data_extender_args=data_extender_args)
	Y = data["label"]
	X = data["text"]

	count_vect = CountVectorizer(min_df=1, ngram_range=(1, 3), analyzer='word', max_features=300)
	X_counts = count_vect.fit_transform(X)
	tfidf_transformer = TfidfTransformer()
	X_tfidf = tfidf_transformer.fit_transform(X_counts)
	classif=SVC()
	search = GridSearchCV(estimator=classif, n_jobs=24, param_grid=parameter_candidates,
						  cv=iterable_indexes)
	# Train the classifier on data1's feature and target data
	search.fit(X_tfidf,Y)
	return [search.cv_results_['mean_test_score'], search.best_estimator_]


def main(is_aug, rounds=100):
	num_classes = 2
	num_folds = 5
	data_extender_args = {
		"iter_num": 10,
		"add_ratio_per_iter": 0.1,
		"diff_threshold_ratio": 0.01 * num_classes,
		"early_stop_threshold": 3,
		"try_num_limits": 150,
		"add_num_decay_rate": 0.5,
		"add_num_decay_method": 'triggered',
		"add_num_decay_stage": None
	}
	whole_dst = get_dataset(dst_name='wheat_corn_reuters_nosplit', model_name="distilbert-base-uncased")

	#set up possible parameters
	parameter_candidates = [{'C':[1,5,10,50,100,500,1000,5000,10000],'gamma':[0.05*x for x in range(1,10)]}]

	results = []
	# different partitions for each round of tuning
	for r in tqdm(range(0, rounds)):
		experiments = crossvalidate(whole_dst, parameter_candidates, num_classes, num_folds, r, data_extender_args, is_aug)
		y = []
		for i in range(0, len(experiments[0])):
			y.append(experiments[0][i])
		results.append(y)
	import pickle

	if is_aug:
		save_name = GS_SAVE_PATH+'aug_REUTERSSVM_differentpartitions_'+str(rounds)+'_'+str(num_folds)+'_folds'
	else:
		save_name = GS_SAVE_PATH+'REUTERSSVM_differentpartitions_'+str(rounds)+'_'+str(num_folds)+'_folds'

	with open(save_name, 'wb') as fp:
		pickle.dump(results, fp)


def repeatedchoicesplotter(R, K, rounds, prefix=''):
	# input R: is nuber of repetitions R (defined as J in paper)
	# input K: is number of folds (either 5,10)
	random.seed(rs)
	values = []
	# load data
	# 1000 independent prediction error estiamtes for each parameter value in the grid
	if (K == 5):
		with open(GS_SAVE_PATH+prefix+'REUTERSSVM_differentpartitions_'+str(rounds*R)+'_5_folds', 'rb') as fp:
			results = pickle.load(fp)
	else:
		print("no precomputed data")
		return None
	# group K-fold CV into groups of R to allow R-K-fold CV
	for i in range(0, math.floor(rounds / R) - 1):
		means = list(results[i * R])
		# average across R K-fold CV grid searches
		for j in range(1, R):
			new = results[i * R + j]
			# for each param choice
			for k in range(0, len(results[0])):
				means[k] = means[k] + new[k]
		for k in range(0, len(results[0])):
			means[k] = means[k] / R
		values.append(means)
	choicesofC = []
	indexofCchoices = []
	indexofGammachoices = []
	choicesofGamma = []
	Cparams = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
	Gammaparams = [0.05 * x for x in range(1, 10)]
	# tuned values of params are maxima across the grids
	for i in range(0, len(values)):
		choicesofC.append(Cparams[math.floor(np.argmax(values[i]) / 9)])
		choicesofGamma.append(Gammaparams[(np.argmax(values[i]) % 9)])
		indexofCchoices.append(math.floor(np.argmax(values[i]) / 9))
		indexofGammachoices.append(np.argmax(values[i]) % 9)
	repeatedscores = []
	for i in range(0, len(values)):
		repeatedscores.append((np.max(values[i])))

	# find range in chosen params
	print("range of chosen C: " + str(min(choicesofC)) + "-" + str(max(choicesofC)))
	print("range of chosen Gamma: " + str(min(choicesofGamma)) + "-" + str(max(choicesofGamma)))
	# find variance of parameter
	print("Variance of chosen C: " + str(np.var(choicesofC, ddof=1)))
	print("Variance of chosen Gamma: " + str(np.var(choicesofGamma, ddof=1)))
	# find variance in estiamted model performance
	print("variance in estimated performance: " + str(np.var(repeatedscores, ddof=1)))
	print("average in estimated performance: " + str(np.mean(repeatedscores)))

	# make plot
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	hist, xedges, yedges = np.histogram2d(indexofGammachoices, indexofCchoices, bins=[np.arange(10), np.arange(10)],
										  normed=True)
	xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1])
	xpos = xpos.flatten('F')
	ypos = ypos.flatten('F')
	zpos = np.zeros_like(xpos)
	dx = 0.5 * np.ones_like(zpos)
	dy = dx.copy()
	dz = hist.flatten()
	ax.set_xlabel('Gamma', color='blue', labelpad=10)
	ax.set_ylabel('C', color='tab:red')
	ax.set_zlabel('Frequency', color='tab:blue')
	ax.set_zlim(0, 0.5)
	ax.tick_params(colors='tab:blue')
	ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='tab:blue', zsort='average')
	xmarks = [i for i in range(0, 8 + 1, 1)]
	plt.xticks(xmarks)
	plt.yticks(xmarks)
	ax.set_xticklabels([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45], rotation=25, color='blue')
	ax.set_yticklabels([1, 5, 10, 50, 100, 500, 1000, 5000, 10000], rotation=-20, va='center', ha='left',
					   color='tab:red')
	plt.show()

	return [np.sqrt(np.var(choicesofC, ddof=1)), np.sqrt(np.var(choicesofGamma, ddof=1)),
			np.sqrt(np.var(repeatedscores, ddof=1)), choicesofC, choicesofGamma, indexofCchoices, indexofGammachoices]


if __name__ == '__main__':
	# 5fold
	main(is_aug=True, rounds=100)
	main(is_aug=False, rounds=100)
	# 4-5fold
	main(is_aug=True, rounds=400)
	main(is_aug=False, rounds=400)
	# draw results
	print('='*20)
	repeatedchoicesplotter(1, 5, 100)
	print('=' * 20)
	repeatedchoicesplotter(1, 5, 100, prefix='aug_')
	print('=' * 20)
	repeatedchoicesplotter(4, 5, 100)
	print('=' * 20)
	repeatedchoicesplotter(4, 5, 100, prefix='aug_')