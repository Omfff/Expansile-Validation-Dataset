
# def get_args():
#     import argparse
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('--val_method', '-vm', choices=['holdout', 'kfold', 'jkfold', 'aug_holdout', 'aug_kfold', 'aug_jkfold',
#                                                         'random_aug_holdout', 'random_aug_kfold',
#                                                         'coreset_holdout', 'random_coreset_holdout', 'aug_coreset_holdout',
#                                                         'part_coreset_holdout', 'part_random_coreset_holdout', 'part_aug_coreset_holdout'], required=True)
#     parser.add_argument('--k', type=int, required=True)
#     parser.add_argument('--J', type=int, required=True)
#     parser.add_argument('--model',  type=str, choices=['xgb', 'rfc', 'lr'], required=True)
#     parser.add_argument('--result_save_path', type=str, required=True)
#     parser.add_argument('--coreset_val_ratio', type=float, required=False, default=0)
#     args = parser.parse_args()
#     return args


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--val_method', '-vm', choices=['holdout', 'kfold', 'jkfold', 'aug_holdout', 'aug_kfold', 'aug_jkfold',
                                                        'random_aug_holdout', 'random_aug_kfold',
                                                        'coreset_holdout', 'random_coreset_holdout', 'aug_coreset_holdout',
                                                        'part_coreset_holdout', 'part_random_coreset_holdout', 'part_aug_coreset_holdout'])
    parser.add_argument('--k', type=int)
    parser.add_argument('--J', type=int)
    parser.add_argument('--model',  type=str, choices=['xgb', 'rfc', 'lr'])
    parser.add_argument('--result_save_path', type=str)
    parser.add_argument('--coreset_val_ratio', type=float, required=False, default=0)
    args = parser.parse_args()
    return args


def complete_cfg_by_args(cfg, args):
    cfg['val_method'] = args.val_method
    cfg['k'] = args.k
    cfg['repeat_j'] = args.J
    cfg['model'] = args.model
    cfg['result_save_path'] = args.result_save_path
    cfg['coreset_val_ratio'] = args.coreset_val_ratio
    return cfg
