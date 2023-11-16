cd ..
cuda_id=0
dst="CIFAR10"
net="ConvNetD3"
ipc=2
sh_file="run_DC_FreD.sh"
eval_mode="S"
data_path="../data"
save_path="./results"

num_eval=5
Iteration=1000
batch_syn=0 # 0 means no sampling (use entire synthetic dataset)
msz_per_channel=32
lr_freq=1e3
mom_freq=0.5

TAG=""
FLAG="${dst}_${ipc}ipc_${net}#DC_FreD_${msz_per_channel}_${batch_syn}#${Iteration}_${lr_freq}_${mom_freq}#${TAG}"

CUDA_VISIBLE_DEVICES=${cuda_id} python3.8 main_DC_FreD.py \
--dataset ${dst} \
--model ${net} \
--ipc ${ipc} \
--sh_file ${sh_file} \
--eval_mode ${eval_mode} \
--data_path ${data_path} --save_path ${save_path} \
--num_eval ${num_eval} \
--Iteration ${Iteration} \
--batch_syn ${batch_syn} \
--msz_per_channel ${msz_per_channel} \
--lr_freq ${lr_freq} --mom_freq ${mom_freq} \
--FLAG ${FLAG}