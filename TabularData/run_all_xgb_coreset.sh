#!/bin/bash

dataset_name_list=('pageblocks56' 'car_eval18' 'car_eval6' 'bank_marketing10p' 'mushroom10p')
main_list=('pageblocks_eval_main56.py' 'car_eval_main18.py' 'car_eval_main6.py' 'bank_marketing_main.py' 'mushroom_main.py')


val_method_list=('coreset_holdout' 'random_coreset_holdout' 'part_coreset_holdout') #
k_list=(1 1 1)
repeatj_list=(1 1 1)

model_name='xgb'
save_path='./experiment/coreset/'
coreset_val_ratio_list=(0.7)


echo ${#val_method_list[@]}

for(( r=0;r<${#coreset_val_ratio_list[@]};r++))
do
  ratio=$(echo "${coreset_val_ratio_list[${r}]}*100"|bc)
  ratio=${ratio%.*}
  echo $ratio

  for(( i=0;i<${#dataset_name_list[@]};i++))
  do
    dataset_name=${dataset_name_list[i]}
    main_py=${main_list[i]}
    index=0
    while(( ${index}<${#val_method_list[@]} ))
    do
      val_method=${val_method_list[${index}]}
      curr_save_name="${save_path}${dataset_name}_${model_name}_${val_method}_${ratio}p.txt"
      echo ${curr_save_name}
      python ${main_py} -vm ${val_method} --k ${k_list[${index}]} --model ${model_name} --J ${repeatj_list[${index}]} \
      --result_save_path ${curr_save_name} --coreset_val_ratio ${coreset_val_ratio_list[${r}]}
      let index+=1
    done
  done
done