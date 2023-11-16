cd ..
cuda_id=0
dst="CIFAR10"
subset="none"
net="ConvNetD3"
ipc=1
sh_file="run_TM_FreD.sh"
eval_mode="S"
data_path="../data"
save_path="./results"
buffer_path="../buffers"

num_eval=5
Iteration=15000
batch_syn=0         # 0 means no sampling (use entire synthetic dataset)
msz_per_channel=64
lr_freq=1e8
mom_freq=0.5

TAG=""
FLAG="${dst}_${subset}_${ipc}ipc_${net}#TM_FreD_${msz_per_channel}_${batch_syn}#${Iteration}_${lr_freq}_${mom_freq}#${TAG}"

CUDA_VISIBLE_DEVICES=${cuda_id} python3.8 main_TM_FreD.py \
--dataset ${dst} --subset ${subset} \
--model ${net} \
--ipc ${ipc} \
--sh_file ${sh_file} \
--eval_mode ${eval_mode} \
--data_path ${data_path} --save_path ${save_path} --buffer_path ${buffer_path} \
--num_eval ${num_eval} \
--Iteration ${Iteration} \
--batch_syn ${batch_syn} \
--msz_per_channel ${msz_per_channel} \
--lr_freq ${lr_freq} --mom_freq ${mom_freq} \
--FLAG ${FLAG}