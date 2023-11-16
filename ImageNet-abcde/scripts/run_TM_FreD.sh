cd ..
cuda_id=0,1,2,3
dst="imagenet-a"
res=128
net="ConvNet"
depth=5
norm_train="instancenorm"
ipc=1
sh_file="run_TM_FreD.sh"
eval_mode="M"
data_path="../../../../../../data/IMAGENET2012"
save_path="./results"
buffer_path="../../../../../../data/tlsehdgur0/buffers"

num_eval=5
eval_it=500
Iteration=15000
batch_syn=0        # 0 means no sampling (use entire synthetic dataset)
msz_per_channel=2048
lr_freq=1e9
mom_freq=0.5

TAG=""
FLAG="${dst}_${res}_${ipc}ipc_${net}D${depth}#TM_FreD_${msz_per_channel}_${batch_syn}#${Iteration}_${lr_freq}_${mom_freq}#${TAG}"

CUDA_VISIBLE_DEVICES=${cuda_id} python3.8 main_TM_FreD.py \
--dataset ${dst} --res ${res} \
--model ${net} --depth ${depth} --norm_train ${norm_train} \
--ipc ${ipc} \
--sh_file ${sh_file} \
--eval_mode ${eval_mode} \
--data_path ${data_path} --save_path ${save_path} --buffer_path ${buffer_path} \
--num_eval ${num_eval} --eval_it ${eval_it} \
--Iteration ${Iteration} \
--batch_syn ${batch_syn} \
--msz_per_channel ${msz_per_channel} \
--lr_freq ${lr_freq} --mom_freq ${mom_freq} \
--FLAG ${FLAG}