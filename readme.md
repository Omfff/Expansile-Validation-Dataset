# Expansile Validation Dataset (EVD): Towards Resolving The Train/Validation Split Tradeoff

This repository is the official implementation of [Expansile Validation Dataset (EVD): Towards Resolving The Train/Validation Split Tradeoff](). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

<img src="https://tva1.sinaimg.cn/large/e6c9d24ely1h10c79lniuj20k10c0wfm.jpg" alt="image-20220406214255058" style="zoom:80%;" />

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```



## Flow of the Code

Each folder contains four core files(ie., `augmentation.py`, `sample.py`, `data_extender.py`, `feature_distribution.py`), the logical role of the files with the same name under different folders are the same, but they are implemented differently due to the different types of data they act on.  The role of each file is explained below:

1. `augmentation.py`: generation of auxiliary dataset by data augmentations
2. `sample.py`:  coreset operation to generate validation set from the auxiliary dataset
3. `data_extender.py`： validation set iterative expansion
4. `feature_distribution.py`:  calculate feature distribution 

Note that, for the sake of brevity, comments on the code are mainly placed in the files under the `CV` folder.

## Datasets

**For Tabular Data**

All datasets are provided in 'TabularData/datasets/'. And you could also download them from kaggle or UCI. 

**For Text**

Except for the feature extractor, all files are already provided in the corresponding folders. If you want to generate these files from scratch, execute the following commands in order.

```shell
# Configure the path_config.yaml, to create configured path, run:
python untils.py

# To load dataset from nltk and save, run:
python reuters.py

# Change the path of models in nlpaug to your local path, and run
CUDA_VISIBLE_DEVICES=0 python augmentation.py

# To obtain feature extractor, run
CUDA_VISIBLE_DEVICES=0 python feature_extractor.py

# To obtain initial val-set by coreset operation, run
CUDA_VISIBLE_DEVICES=0 python samply.py
```

**For CV**

Due to the large file size of the image dataset, we here only provide the command to obtain these files rather than the original files.

```shell
# Configure the path_config.yaml, to create configured path, run:
python untils.py

# Generate cifar10-longtail
python imbalanced_dataset.py

# To generate auxiliary dataset
python augmentation.py

# To obtain feature extractor, run
CUDA_VISIBLE_DEVICES=0 python feature_extractor.py

# To obtain initial val-set by coreset operation, run
CUDA_VISIBLE_DEVICES=0 python samply.py
```



## Train and Evaluation

To get the results in the paper, run following commands:

**Results of tabular data**

```bash
# Configure the save path in run_all_xgb.sh and run
./run_all_xgb.sh

# Configure the save path in run_all_xgb_coreset.sh and then run
./run_all_xgb_coreset.sh

# Configure the same path as the two above at the beginning of the file
# then run the command to get the statistics in all Tables 
python record.py
```

**Results of  Reuters(Text, NLP)**

```bash
# For results in Tabel 2

CUDA_VISIBLE_DEVICES=0 python reuter_eval_main.py -vm holdout --k 1 --save_name xxx

CUDA_VISIBLE_DEVICES=0 python reuter_eval_main.py -vm kfold --k 5 --save_name xxx

CUDA_VISIBLE_DEVICES=0 python reuter_eval_main.py -vm jkfold --k 5 --J 4 --save_name xxx

CUDA_VISIBLE_DEVICES=0 python reuter_eval_main.py -vm aug_coreset_whole --k 1 --fe_type fine-tune --feature_dis_type NDB --save_name xxx



# For results in Tabel 5

CUDA_VISIBLE_DEVICES=0 python reuter_eval_main.py -vm coreset_part_holdout --k 1 --save_name xxx


# For results in Tabel 6

CUDA_VISIBLE_DEVICES=0 python reuter_eval_main.py -vm aug_holdout --k 1 --fe_type fine-tune --feature_dis_type NDB --save_name xxx

CUDA_VISIBLE_DEVICES=0 python reuter_eval_main.py -vm aug_kfold --k 5 --fe_type fine-tune --feature_dis_type NDB --save_name xxx


# For results in Figure 2

python reuters_hyper_params_search.py
```

**Results of  CIFAR-10-LT(Image, CV)**

```bash
# For results in Tabel 2

CUDA_VISIBLE_DEVICES=0 python cifar10_eval_main.py -vm holdout --k 1 --save-dir xxx

CUDA_VISIBLE_DEVICES=0 python cifar10_eval_main.py -vm kfold --k 5 --save-dir xxx

CUDA_VISIBLE_DEVICES=0 python cifar10_eval_main.py -vm jkfold --J 4 --k 5 --save-dir xxx

CUDA_VISIBLE_DEVICES=0 python cifar10_eval_main.py -vm coreset_whole --k 1 --save-dir xxx


# For results in Tabel 4

CUDA_VISIBLE_DEVICES=0 python cifar10_eval_main.py -vm random_coreset --k 1 --save-dir xxx


# For results in Tabel 5

CUDA_VISIBLE_DEVICES=0 python cifar10_eval_main.py -vm coreset_part_holdout --k 1 --save-dir xxx


# For results in Tabel 6

CUDA_VISIBLE_DEVICES=0 python cifar10_eval_main.py -vm aug_holdout --k 1 --feature_dis_type NDB --config_path ./config/cifar10_default.yaml --save-dir xxx

CUDA_VISIBLE_DEVICES=0 python cifar10_eval_main.py -vm aug_kfold --k 5 --feature_dis_type NDB --config_path ./config/cifar10_default.yaml --save-dir xxx
```



## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 



