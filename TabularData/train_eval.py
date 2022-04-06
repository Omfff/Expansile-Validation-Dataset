import copy
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from feature_distribution import FeatureDistribution,FeatureType
from augmentation import DataGenerator
from data_extender import DataExtender
from util import merge_dict, normalize_features, cal_value_list, print_result
from coreset import CoresetSampler, RandomCoresetSampler


def train(cfg, seed_set, train_set, test_set, feature_cols:FeatureType, label_list):
    perf_dict = {
        'val_f1_score_mean': [],
        'val_auc_mean': [],
        'test_f1_score_mean': [],
        'test_auc_mean': [],
        'val_test_bias_list': [],
        'val_test_bias_mean': [],
        'f1_score_group_std': [],
        'auc_group_std': []
    }

    if 'jkfold' in cfg['val_method']:
        J = cfg['repeat_j']
        for seed_index in range(len(seed_set)):
            print(seed_index)
            np.random.seed(seed_set[seed_index])
            kfold_seeds = np.random.randint(0, 10000, size=J)
            jkfold_perf_dict = {'val_f1_score': [],
                                'val_auc': [],
                                'val_test_bias': [],
                                'test_f1_score': [],
                                'test_auc': []}
            for j in range(J):
                performance_dic = one_round_training(cfg, kfold_seeds[j], train_set, test_set, feature_cols,
                                                                  label_list)
                jkfold_perf_dict['val_f1_score'].append(np.asarray(performance_dic['val_f1_score']).mean())
                jkfold_perf_dict['val_auc'].append(np.asarray(performance_dic['val_auc']).mean())
                jkfold_perf_dict['test_f1_score'].append(np.asarray(performance_dic['test_f1_score']).mean())
                jkfold_perf_dict['test_auc'].append(np.asarray(performance_dic['test_auc']).mean())
                jkfold_perf_dict['val_test_bias'].append(np.asarray(performance_dic['val_test_bias']).mean())

            perf_dict['val_f1_score_mean'].append(np.asarray(jkfold_perf_dict['val_f1_score']).mean())
            perf_dict['val_auc_mean'].append(np.asarray(jkfold_perf_dict['val_auc']).mean())
            perf_dict['test_f1_score_mean'].append(np.asarray(jkfold_perf_dict['test_f1_score']).mean())
            perf_dict['test_auc_mean'].append(np.asarray(jkfold_perf_dict['test_auc']).mean())
            perf_dict['val_test_bias_mean'].append(np.asarray(jkfold_perf_dict['val_test_bias']).mean())

    else:
        for seed_index in range(len(seed_set)):
            print(seed_index)
            performance_dic = one_round_training(cfg, seed_set[seed_index], train_set, test_set, feature_cols, label_list)
            if cfg['cal_distribution']:
                continue

            perf_dict['val_f1_score_mean'].append(np.asarray(performance_dic['val_f1_score']).mean())
            perf_dict['val_auc_mean'].append(np.asarray(performance_dic['val_auc']).mean())
            perf_dict['test_f1_score_mean'].append(np.asarray(performance_dic['test_f1_score']).mean())
            perf_dict['test_auc_mean'].append(np.asarray(performance_dic['test_auc']).mean())
            perf_dict['val_test_bias_mean'].append(np.asarray(performance_dic['val_test_bias']).mean())
            # print(perf_dict)
            # perf_dict['val_test_bias_list'].extend(performance_dic['val_test_bias'])

    # print(perf_dict)
    final_result = print_result(perf_dict)
    with open(cfg['result_save_path'], 'wb') as f:
        pickle.dump(final_result, f)
    return perf_dict, final_result


def one_round_training(cfg, seed, train_set, test_set, feature_cols:FeatureType, label_list):
    f_count = 0
    feature_dis = {}
    model = cfg['model']
    test_set_copy = copy.deepcopy(test_set)

    train_set = (train_set[0].reset_index(drop=True), train_set[1].reset_index(drop=True))

    if 'kfold' in cfg['val_method']:
        k = cfg['k']
        kfold = StratifiedKFold(n_splits=cfg['k'], shuffle=True, random_state=seed)
        kfold_set = []
        for train_ind, val_ind in kfold.split(train_set[0], train_set[1]):
            kfold_set.append((train_ind, val_ind))
    else:
        k = 1

    performance_dic = {
        'val_f1_score': [],
        'val_auc': [],
        'val_test_bias': [],
        'test_f1_score': [],
        'test_auc': []
    }

    for i in range(k):
        if k > 1:
            init_x_train = train_set[0].iloc[kfold_set[i][0].tolist()]
            init_y_train = train_set[1].iloc[kfold_set[i][0].tolist()]
            init_x_val = train_set[0].iloc[kfold_set[i][1].tolist()]
            init_y_val = train_set[1].iloc[kfold_set[i][1].tolist()]
        else:
            init_x_train, init_x_val, init_y_train, init_y_val = train_test_split(train_set[0], train_set[1],
                                                            test_size=cfg['val_ratio'],
                                                            random_state=seed,
                                                            stratify=train_set[1])
            if 'coreset' in cfg['val_method']:
                d_generator = DataGenerator(train_set, random_seed=seed,
                                            categorical_col=feature_cols.get_categorical_cols())
                data_pool = d_generator.merge_set(d_generator.generate_data_to_pool(method=cfg['aug_method']))
                if 'random' in cfg['val_method']:
                    sampler = RandomCoresetSampler(whole_train_set=train_set, data_pool=data_pool, random_seed=seed)
                else:
                    sampler = CoresetSampler(whole_train_set=train_set, data_pool=data_pool, random_seed=seed)

                val_indx = sampler.coreset_sample(val_ratio=cfg['coreset_val_ratio'])
                init_x_val = data_pool[0].iloc[val_indx]
                init_y_val = data_pool[1].iloc[val_indx]
                if 'part' in cfg['val_method']:
                    pass
                else:
                    init_x_train = train_set[0]
                    init_y_train = train_set[1]

        init_train_set = (init_x_train.reset_index(drop=True), init_y_train.reset_index(drop=True))
        init_val_set = (init_x_val.reset_index(drop=True), init_y_val.reset_index(drop=True))
        if cfg['cal_distribution']:
            fd = FeatureDistribution(None, None, labels=label_list,
                                     discrete_col=feature_cols.get_discrete_cols(),
                                     numerical_col=feature_cols.get_numerical_cols())
            whole_train_distribution = fd.cal_distribution(train_set[0], train_set[1])
            val_distribution = fd.cal_distribution(init_val_set[0], init_val_set[1])
            feature_dis = merge_dict(fd.cal_distribution_diff(whole_train_distribution, val_distribution), feature_dis)
            f_count += 1
            continue

        if 'aug' in cfg['val_method']:
            fd = FeatureDistribution(train_set[0], train_set[1], labels=label_list,
                                     discrete_col=feature_cols.get_discrete_cols(),
                                     numerical_col=feature_cols.get_numerical_cols())

            aug_set = DataExtender((train_set[0], train_set[1]), init_train_set, init_val_set,
                                 fd=fd, iter_num=cfg['val_aug_iter_num'], add_ratio_per_iter=cfg['add_ratio_per_iter'],
                                 diff_threshold_ratio=cfg['diff_threshold_ratio'], early_stop_threshold=cfg["early_stop_threshold"],
                                 try_num_limits=cfg["try_num_limits"], add_num_decay_rate=cfg["add_num_decay_rate"],
                                 add_num_decay_method=cfg["add_num_decay_method"], add_num_decay_stage=cfg["add_num_decay_stage"],
                                 random_seed=seed,
                                 categorical_col=feature_cols.get_categorical_cols())
            aug_set.generate_data_to_pool(method=cfg['aug_method'])
            if 'random' in cfg['val_method']:
                (x_train, y_train), (x_val, y_val) = aug_set.run_random_select_without_limitation()
            else:
                (x_train, y_train), (x_val, y_val) = aug_set.run(sample_method=cfg['aug_sample_method'])
        else:
            (x_train, y_train), (x_val, y_val)= init_train_set, init_val_set

        if cfg['model'] == 'lr':
            value_list = cal_value_list(x_train, feature_cols.get_categorical_cols())
            x_train = normalize_features(feature_cols, x_train, value_list)
            x_val = normalize_features(feature_cols, x_val, value_list)
            x_test = normalize_features(feature_cols, test_set_copy[0], value_list)
            test_set = (x_test, test_set[1])

        print('train set shape', len(x_train))
        print('val set shape', len(x_val))

        x_train = x_train.values
        y_train = y_train.values
        x_val = x_val.values
        y_val = y_val.values

        print('training .....')
        if model == 'rfc':
            cls = RandomForestClassifier(random_state=0, n_jobs=cfg['model_params']['n_jobs'])
            cls.fit(x_train, y_train)
            # Predict the Test set results
        elif model == 'xgb':
            cls = XGBClassifier(**cfg['model_params'])
            # cls.fit(x_train, y_train, eval_metric='auc', verbose=False)
            cls.fit(x_train, y_train, eval_metric=['auc'], eval_set=[(x_val, y_val)], verbose=False, early_stopping_rounds=20)
        elif model == 'lr':
            cls = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=1000, n_jobs=32)
            cls.fit(x_train, y_train)

        y_test_pred = cls.predict(test_set[0].values)
        y_val_pred = cls.predict(x_val)

        f1_val = f1_score(y_val, y_val_pred)
        f1_test = f1_score(test_set[1].values, y_test_pred)

        y_val_p = cls.predict_proba(x_val)
        y_val_pred_pos = y_val_p[:, 1]
        y_test_p = cls.predict_proba(test_set[0])
        y_test_pred_pos = y_test_p[:, 1]
        auc_val = roc_auc_score(y_val, y_val_pred_pos)
        auc_test = roc_auc_score(test_set[1].values, y_test_pred_pos)

        performance_dic['val_f1_score'].append(f1_val)
        performance_dic['val_auc'].append(auc_val)
        performance_dic['val_test_bias'].append(abs(f1_val - f1_test))
        performance_dic['test_f1_score'].append(f1_test)
        performance_dic['test_auc'].append(auc_test)
        # print(performance_dic)

    return performance_dic

