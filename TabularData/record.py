import pickle
from draw_result import draw1, draw2


def load_file(file_name):
    with open(file_name, 'rb') as f:
        record = pickle.load(f)
    return record


def read_one_record(model_name, dataset_name, method, file_folder):
    records = {}
    for m in method:
        if 'coreset' in m:
            file_path = file_folder+'coreset/' + dataset_name + '_' + model_name + '_' + m + '.txt'
        else:
            # file_path = file_folder+ 'reproduce3/' +dataset_name+'_'+model_name+'_'+m+'.txt'
            file_path = file_folder+dataset_name+'_'+model_name+'_'+m+'.txt'
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
    file_folder = './experiments/'
    # results in table 2
    results2 = baseline_vs_coreset(model_list, dataset_name_list, file_folder = './experiments/early_stop/')
    output_for_latex(results2)
    # results in table 4
    results4 = random_coreset_vs_coreset(model_list, dataset_name_list, file_folder = './experiments/early_stop/')
    output_for_latex(results4)
    # results in table 5
    results5 = holdout_vs_part_coreset(model_list, dataset_name_list, file_folder = './experiments/early_stop/')
    output_for_latex(results5)
    # results in table 6
    results6 = baseline_vs_expansion(model_list, dataset_name_list, file_folder='./experiments/early_stop/')
    output_for_latex(results6)

