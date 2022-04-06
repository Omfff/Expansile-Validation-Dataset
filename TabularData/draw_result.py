from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np


def fd_diff_std_relation():
    feature_distribution_diff0 = [0.006, 0.013, 0.002, 0.004, 0.022, 0.012, 0.006, 0.002, 0.002]
    feature_distribution_diff1 = [0.006, 0.010, 0.005, 0.011, 0.055, 0.064, 0.007, 0.016, 0.052]
    feature_distribution_diff0_js = [0.027, 0.078, 0.011, 0.034, 0.057, 0.037, 0.114, 0.033, 0.033]
    feature_distribution_diff1_js = [0.023, 0.064, 0.028, 0.090, 0.12,  0.12, 0.114, 0.191, 0.399]
    std_auc = [0, 0.0018, 0.001, 0.008, 0.014, 0.006, 0.002, 0.009, 0.036]
    std_f1 = [0, 0.0018, 0.003, 0.020, 0.029, 0.011, 0.003, 0.020, 0.052]
    result1 = spearmanr(feature_distribution_diff1, std_auc)
    result2 = spearmanr(feature_distribution_diff1, std_f1)
    result3 = spearmanr(feature_distribution_diff0, std_auc)
    result4 = spearmanr(feature_distribution_diff0, std_f1)

    result1_js = spearmanr(feature_distribution_diff1_js, std_f1)
    result0_js = spearmanr(feature_distribution_diff0_js, std_f1)
    result2_js = spearmanr(feature_distribution_diff0_js, std_auc)

    print(result1)
    print(result2)
    print(result3)
    print(result4)
    print(result1_js)
    print(result0_js)
    print(result2_js)


def draw_coreset_holdout_kfold_std_and_bias():

    holdout_std = [0.125 ,0.046 ,0.046 ,0.079 ,0.059 ,0.008 ,0.0006 ,0.021]
    holdout_coreset_std = [0.003, 0.019, 0.014, 0.044, 0.030, 0.004, 0.0000, 0.008]
    kfold_std = [0.046, 0.010, 0.009, 0.020, 0.016, 0.002, 0.0000, 0.004]

    holdout_bias = [0.140,0.041,0.035,0.063,0.071,0.0056,0.0008,0.033]
    holdout_coreset_bias = [0.507,0.061,0.078,0.086,0.042,0.019,0.0006,0.082]
    kfold_bias = [0.161,0.038,0.031,0.056,0.077,0.006,0.0008,0.031]

    kfold_fdd0 = [0.0026, 0.0026, 0.013, 0.0229, 0.0056, 0.0127, 0.0054, 0.0025]
    kfold_fdd1 = [0.055, 0.016, 0.0642, 0.063, 0.011, 0.013, 0.0056, 0.005]
    holdout_fdd0 = [0.0025, 0.0025, 0.0127, 0.0232, 0.0056, 0.0125, 0.0053, 0.0025]
    holdout_fdd1 = [0.058, 0.015, 0.065, 0.063, 0.011, 0.013, 0.0056, 0.0049]
    holdout_coreset_fdd0 = [0.009, 0.008, 0.019, 0.016, 0.0144, 0.0147, 0.0153, 0.0186]
    holdout_coreset_fdd1 = [0.0449, 0.0456, 0.0478, 0.0518, 0.0159, 0.0281, 0.026, 0.016]
    print("kfold fdd0 ~ std", spearmanr(kfold_fdd0, kfold_std))
    print("kfold fdd1 ~ std", spearmanr(kfold_fdd1, kfold_std))
    print("kfold fdd0 ~ bias", spearmanr(kfold_fdd0, kfold_bias))
    print("kfold fdd1 ~ bias", spearmanr(kfold_fdd1, kfold_bias))
    print("holdout fdd0 ~ std", spearmanr(holdout_fdd0, holdout_std))
    print("holdout fdd1 ~ std", spearmanr(holdout_fdd1, holdout_std))
    print("holdout fdd0 ~ bias", spearmanr(holdout_fdd0, holdout_bias))
    print("holdout fdd1 ~ bias", spearmanr(holdout_fdd1, holdout_bias))
    print("holdout_coreset fdd0 ~ std", spearmanr(holdout_coreset_fdd0, holdout_coreset_std))
    print("holdout_coreset fdd1 ~ std", spearmanr(holdout_coreset_fdd1, holdout_coreset_std))
    print("holdout_coreset fdd0 ~ bias", spearmanr(holdout_coreset_fdd0, holdout_coreset_bias))
    print("holdout_coreset fdd1 ~ bias", spearmanr(holdout_coreset_fdd1, holdout_coreset_bias))
    print("holdout_coreset  ~ std", spearmanr(holdout_std+holdout_coreset_std, holdout_fdd1+holdout_coreset_fdd1))
    #'CreditCard',
    x_axis_name = ['PageBlock(175:1)' ,'PageBlock(42:1)', 'CarEval(18:1)', 'CarEval(6:1)', 'BankMarket(10%)', 'MushRoom(10%)', 'MushRoom', 'BankMarket']
    x_axis = [i for i in range(len(x_axis_name))]
    fig, ax = plt.subplots(1,2, figsize=(14,5))
    ax[0].plot(holdout_std, label='holdout')
    ax[0].plot(holdout_coreset_std, label='holdout_coreset')
    ax[0].plot(kfold_std, label='5fold')
    ax[0].set_title('Variance')
    ax[0].set_xticks(x_axis)
    ax[0].set_xticklabels(x_axis_name, rotation=20, fontsize=8)
    ax[0].legend()
    ax[1].plot(holdout_bias, label='holdout')
    ax[1].plot(holdout_coreset_bias, label='holdout_coreset')
    ax[1].plot(kfold_bias, label='5fold')
    ax[1].set_title('Bias')
    ax[1].set_xticks(x_axis, )
    ax[1].set_xticklabels(x_axis_name, rotation=20, fontsize=8)
    ax[1].legend()
    plt.show()


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


def draw_aug_holdout_kfold_std_and_bias():
    holdout_std = [0.125, 0.046, 0.046, 0.079]
    holdout_coreset_std = [0.003, 0.019, 0.014, 0.044]
    holdout_aug_std = [0.058, 0.024, 0.020, 0.030]
    kfold_std = [0.046, 0.010, 0.0092, 0.020]
    kfold_aug_std = [0.025, 0.009, 0.0094, 0.014]

    holdout_bias = [0.140, 0.041, 0.035, 0.063]
    holdout_coreset_bias = [0.507, 0.061, 0.078, 0.080]
    holdout_aug_bias = [0.165, 0.041, 0.017, 0.024]
    kfold_bias = [ 0.161, 0.038, 0.031, 0.056]
    kfold_aug_bias = [0.178, 0.039, 0.020, 0.028]


    #'CreditCard',
    x_axis_name = ['PageBlock(175:1)' ,'PageBlock(42:1)', 'CarEval(18:1)', 'CarEval(6:1)']


if __name__ == '__main__':
    draw_aug_holdout_kfold_std_and_bias()

