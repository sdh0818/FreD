cd ..
cuda_id=0
dst="imagenet-a"
res=128
net="ConvNet"
depth=5
ipc=1
sh_file="run_DC_FreD.sh"
eval_mode="M"
data_path="../../../../../../data/IMAGENET2012"
save_path="./results"

num_eval=5
Iteration=1000
batch_syn=0 # 0 means no sampling (use entire synthetic dataset)
msz_per_channel=2048
lr_freq=1e5
mom_freq=0.5

TAG=""
FLAG="${dst}_${res}_${ipc}ipc_${net}D${depth}#DC_FreD_${msz_per_channel}_${batch_syn}#${Iteration}_${lr_freq}_${mom_freq}#${TAG}"

CUDA_VISIBLE_DEVICES=${cuda_id} python3.8 main_DC_FreD.py \
--dataset ${dst} --res ${res} \
--model ${net} --depth ${depth} \
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