#!/bin/bash
# cd ../../src/3_evaluate/
# to_ex_path=eval_CoFaithfulQA.py
# eval_files= 'your file path'
# to_eval_model=$1
# output_dir=$2
# device=$3
# schema=$4
# max_new_tokens=$5
# act_inhibit_ratio=$6
# echo "using $device"



# # 测试一个模型
# cur_model=${to_eval_model}
# for file in ${eval_files}/*.jsonl
# do
#     filename=$(basename "$file")

#     echo "###################### eval ${to_eval_model} ######################"
#     filename=$(basename "$file")
#     echo "###################### infer ${filename} ######################"
#     CUDA_VISIBLE_DEVICES=$device python3 $to_ex_path \
#         --model_name $cur_model \
#         --data_path $file \
#         --schema $schema \
#         --output_path ${output_dir}/${filename}_res.json \
#         --log_path ${output_dir}/${filename}.log \
#         --use_chat_template $use_chat_template \
#         --max_new_tokens $max_new_tokens \
#         --act_inhibit_ratio $act_inhibit_ratio \
#         --act_inhibit_layer_list 19 20 21 22 23 24 25 26
   
# done


# to_ex_path=./src/3_evaluate/eval_CoConflictQA.py
# eval_files="./data/CoConflictQA/test"
# to_eval_model=$1
# output_dir=$2
# device=$3
# schema=$4       # base instr+opin attr and so on ... line190 in eval_CoFaithfulQA.py
# max_new_tokens=$5
# act_inhibit_ratio=$6
# echo "==============================="
# echo "🚀 Using device: $device"
# echo "==============================="


# # 测试一个模型
# cur_model=${to_eval_model}
# for file in ${eval_files}/*.jsonl
# do
#     filename=$(basename "$file")

#     echo "###################### eval ${to_eval_model} ######################"
#     filename=$(basename "$file")
#     echo "###################### infer ${filename} ######################"
#     CUDA_VISIBLE_DEVICES=$device python3 $to_ex_path \
#         --model_name $cur_model \
#         --data_path $file \
#         --schema $schema \
#         --output_path ${output_dir}/${filename}_res.json \
#         --log_path ${output_dir}/${filename}.log \
#         --use_chat_template $use_chat_template \
#         --max_new_tokens $max_new_tokens \
#         --act_inhibit_ratio $act_inhibit_ratio \
#         --act_inhibit_layer_list 21 22 23 24 25 26
# done
# find the most suitable act_inhibit_layer_list with /scripts/1_Identifying/1_visualize.sh


to_ex_path=./src/3_evaluate/eval_CoConflictQA.py
eval_files="./data/CoConflictQA/test"
to_eval_model=$1
output_dir=$2
device=$3
schema=$4       # base instr+opin attr and so on ... line190 in eval_CoFaithfulQA.py
max_new_tokens=$5
act_inhibit_ratio=$6
use_chat_template=$7
echo "==============================="
echo "🚀 Using device: $device"
echo "==============================="

# 测试一个模型
cur_model=${to_eval_model}
for file in ${eval_files}/*.jsonl
do
    filename=$(basename "$file")

    echo "###################### eval ${to_eval_model} ######################"
    filename=$(basename "$file")
    echo "###################### infer ${filename} ######################"
    CUDA_VISIBLE_DEVICES=$device python3 $to_ex_path \
        --model_name $cur_model \
        --data_path $file \
        --schema $schema \
        --output_path ${output_dir}/${filename}_res.json \
        --log_path ${output_dir}/${filename}.log \
        --use_chat_template $use_chat_template \
        --max_new_tokens $max_new_tokens \
        --act_inhibit_ratio $act_inhibit_ratio \
        --act_inhibit_layer_list 21 22 23 24 25 26
done
