import pickle
import matplotlib.pyplot as plt
import numpy as np

EXPERIMENTS_SAVE_PATH = './experiments/'
CORSET_EXPAN_SAVE_FOLDER = 'coreset'
BASELINE_SAVE_FOLDER = 'baseline'


def draw1(results, dst_list, method_list, line_style_list, line_color_list, labels=None):
    num_sub_fig = 3
    x_axis_name = dst_list
    x_axis = [i for i in range(len(x_axis_name))]
    label_list = method_list if labels is None else labels
    fig, ax = plt.subplots(1, num_sub_fig, figsize=(23, 5))
    for i, me in enumerate(method_list):
        ax[0].plot(results[me+'_std'], label=label_list[i], linestyle=line_style_list[i], color=line_color_list[i])
        ax[1].plot(results[me+'_bias'], label=label_list[i], linestyle=line_style_list[i], color=line_color_list[i])
        ax[2].plot(results[me+'_test_f1'], label=label_list[i], linestyle=line_style_list[i], color=line_color_list[i])

    title_list = ['Variance', 'Bias', 'Test F1']
    for i in range(num_sub_fig):
        ax[i].set_title(title_list[i])
        ax[i].set_xticks(x_axis)
        ax[i].set_xticklabels(x_axis_name, rotation=20, fontsize=8)
    ax[num_sub_fig-1].legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    plt.show()


def draw2(results, dst_list, method_list, color_list, hatch_list, labels=None):
    x_axis_name = dst_list
    x_axis = np.arange(len(x_axis_name))

    num_subplots = 3

    total_width, n = 1, len(method_list)+1
    width = total_width / n
    x_axis = x_axis - (total_width - width) / 2

    label_list = method_list if labels is None else labels

    fig, ax = plt.subplots(1, num_subplots, figsize=(40, 5))
    for i,m in enumerate(method_list):
        ax[0].bar(x_axis + i * width, results[m + '_std'], width=width, label=label_list[i], color=color_list[i], hatch=hatch_list[i])
        ax[1].bar(x_axis + i * width, results[m + '_bias'], width=width, label=label_list[i], color=color_list[i], hatch=hatch_list[i])
        ax[2].bar(x_axis + i * width, results[m + '_test_f1'], width=width, label=label_list[i], color=color_list[i], hatch=hatch_list[i])

    title_list = ['Variance', 'Bias', 'F1']

    for j in range(num_subplots):
        ax[j].set_title(title_list[j])
        ax[j].set_xticks(x_axis)
        ax[j].set_xticklabels(x_axis_name, rotation=20, fontsize=8)

    ax[-1].legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    plt.subplots_adjust(left=0.02)
    plt.show()


def load_file(file_name):
    with open(file_name, 'rb') as f:
        record = pickle.load(f)
    return record


def read_one_record(model_name, dataset_name, method, file_folder):
    records = {}
    for m in method:
        if 'coreset' in m:
            file_path = file_folder + CORSET_EXPAN_SAVE_FOLDER +'/' + dataset_name + '_' + model_name + '_' + m + '.txt'
        else:
            file_path = file_folder+ BASELINE_SAVE_FOLDER +'/' +dataset_name+'_'+model_name+'_'+m+'.txt'
        r = load_file(file_path)
        print(r)
        r['std'] = r.pop('val_f1_score_mean_std')
        r['bias'] =r.pop('val_test_bias_mean_mean')
        r['test_f1'] = r.pop('test_f1_score_mean_mean')
        r.pop('val_auc_mean_std')
        r.pop('test_auc_mean_mean')
        records[m] = r

    return records


def convert_to_draw_format(results_dict, methods):
    list_results= {}
    for model in results_dict.keys():
        list_results[model] = {}
        for m in methods:
            list_results[model][m+'_std'] = []
            list_results[model][m+'_bias'] = []
            list_results[model][m+'_test_f1'] = []
        for dst,r in results_dict[model].items():
            for method,v in r.items():
                list_results[model][method+'_std'].append(v['std'])
                list_results[model][method+'_bias'].append(v['bias'])
                list_results[model][method+'_test_f1'].append(v['test_f1'])
    return list_results


def read_and_draw(dataset_name_list, model_list, methods, file_folder,
                  line_style_list, hatch_list, color_list):
    all_results = {}
    for m in model_list:
        all_results[m] = {}
        for dst_name in dataset_name_list:
            records = read_one_record(m, dst_name, methods, file_folder)
            all_results[m][dst_name] = records
    print(all_results)
    results_for_draw = convert_to_draw_format(all_results, methods)
    print(results_for_draw)

    for m in model_list:
        draw2(results_for_draw[m], dataset_name_list, methods, color_list, hatch_list, labels=None)
        draw1(results_for_draw[m], dataset_name_list, methods, line_style_list, color_list, labels=None)

    return all_results


def baseline_vs_coreset(model_list, dataset_name_list, file_folder = './experiments/'):
    import matplotlib.pyplot as plt
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    methods = ['holdout', 'kfold', 'jkfold', 'coreset_holdout_70p', 'aug_coreset_holdout_70p']
    line_style_list = ['--', '--', '--','-','-']
    hatch_list = ['', '', '', '//', '//']
    color_list = [cycle[0], cycle[1], cycle[2], cycle[3], cycle[4]]  # line_color_list
    return read_and_draw(dataset_name_list, model_list, methods, file_folder,
                  line_style_list, hatch_list, color_list)


def random_coreset_vs_coreset(model_list, dataset_name_list, file_folder='./experiments/'):
    import matplotlib.pyplot as plt
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    methods = ['random_coreset_holdout_70p', 'coreset_holdout_70p', 'aug_coreset_holdout_70p']
    line_style_list = [':', '--', '-']
    hatch_list = ['', '', '']
    color_list = [cycle[0], cycle[0], cycle[0]]  # line_color_list
    return read_and_draw(dataset_name_list, model_list, methods, file_folder, line_style_list, hatch_list, color_list)


def holdout_vs_part_coreset(model_list, dataset_name_list, file_folder = './experiments/'):
    import matplotlib.pyplot as plt
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    methods = ['holdout',  'part_coreset_holdout_70p', 'coreset_holdout_70p']
    line_style_list = ['--', '-', '-']
    hatch_list = ['', '.', '//']
    color_list = [cycle[0], cycle[1], cycle[2]]  # line_color_list
    return read_and_draw(dataset_name_list, model_list, methods, file_folder,
                  line_style_list, hatch_list, color_list)


def baseline_vs_expansion(model_list, dataset_name_list, file_folder = './experiments/'):
    import matplotlib.pyplot as plt
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    methods = ['holdout', 'aug_holdout',
               'kfold',  'aug_kfold']
    line_style_list = ['--', '-',
                       '--',  '-']
    hatch_list = ['', '//',
                  '', '//']
    color_list = [cycle[0], cycle[0], cycle[1], cycle[1]]  # line_color_list

    return read_and_draw(dataset_name_list, model_list, methods, file_folder,
                  line_style_list, hatch_list, color_list)


def output_for_latex(data):
    for m, dst_dict in data.items():
        for dst, method_dict in dst_dict.items():
            print('='*20 + dst + 20*'=')
            for me, metrics_dict in method_dict.items():
                print(me)
                for met, value in metrics_dict.items():
                    print(round(value, 4), end=' & ')
                print()


if __name__ == '__main__':
    model_list = ['xgb']
    dataset_name_list = ['pageblocks42', 'car_eval18', 'car_eval6', 'bank_marketing10p', 'mushroom10p']
    # results in table 2
    results2 = baseline_vs_coreset(model_list, dataset_name_list, file_folder=EXPERIMENTS_SAVE_PATH)
    output_for_latex(results2)
    # results in table 4
    results4 = random_coreset_vs_coreset(model_list, dataset_name_list, file_folder=EXPERIMENTS_SAVE_PATH)
    output_for_latex(results4)
    # results in table 5
    results5 = holdout_vs_part_coreset(model_list, dataset_name_list, file_folder=EXPERIMENTS_SAVE_PATH)
    output_for_latex(results5)
    # results in table 6
    results6 = baseline_vs_expansion(model_list, dataset_name_list, file_folder=EXPERIMENTS_SAVE_PATH)
    output_for_latex(results6)

