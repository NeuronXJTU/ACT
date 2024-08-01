#!/bin/bash

# ulimit -SHn 65535
# ulimit -a

net1=ResGCN14
net2=EfficientNetB0
net3=pvig_ti_224_gelu
image_txt="./txts/CRC_datasets.txt"
test_txt="./txts/pt_CRC_new_test.txt"
train_txt="./txts/pt_CRC_new_train.txt"
num_classes=3
num_net=2
# resume_gcn=""
# resume_cnn=""

log_dir="./logs/dml_2/CRC_new_${net1}_${net2}"
output_dir="./results/dml_2/CRC_new_${net1}_${net2}"


if ! [ -d ${log_dir} ];then
    mkdir -p ${log_dir}
fi

if ! [ -d ${output_dir} ];then
    mkdir -p ${output_dir}
fi

CUDA_VISIBLE_DEVICES=4 \
nohup python -u train.py \
             --image_txt ${image_txt} \
             --valid_patient_txts ${test_txt}\
             --patient_txts ${train_txt} \
             --num_classes ${num_classes} \
             --net1 ${net1} \
             --net2 ${net2} \
             --net3 ${net3} \
             --num_net ${num_net} \
             --resume_gcn "" \
             --resume_cnn "" \
             --epochs 200 \
             --batch_size 32 \
             --world_size 1 \
             --num_workers 4 \
             --kld 0.10 \
             --smoothing 0.1 \
             --optim AdamW \
             --log_dir ${log_dir} \
             --output_dir ${output_dir} \
             > ${log_dir}/nohup.log 2>&1 &