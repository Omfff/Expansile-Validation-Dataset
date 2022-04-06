
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--val_method', '-vm', choices=['holdout', 'kfold', 'aug_holdout', 'aug_kfold', 'jkfold',
                                                        'coreset_part_holdout', 'coreset_whole', 'aug_coreset_whole'],
                        required=True)
    parser.add_argument('--J', type=int, default=1, required=False)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--fe_type',  type=str, choices=['fine-tune', 'pre-train'], required=False)
    parser.add_argument('--feature_dis_type', type=str, choices=['CD', 'NDB'], required=False)
    parser.add_argument('--save_name', type=str, required=False, default='experiments/')
    parser.add_argument('--ignore_fdd', dest='ignore_fdd', action='store_true')
    parser.add_argument('--syn_val', dest='syn_val', action='store_true')

    args = parser.parse_args()
    ignore_fdd = '_ignore_fdd' if args.ignore_fdd else ''
    fe_type = '' if args.fe_type is None else args.fe_type
    feature_dis_type = '' if args.feature_dis_type is None else args.feature_dis_type
    args.save_name = args.save_name + args.val_method + '_' + fe_type + '_' + feature_dis_type + ignore_fdd + '.pkl'
    return args

