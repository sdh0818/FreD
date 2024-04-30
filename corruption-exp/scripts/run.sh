cd ..
cuda_id=2
method="FreD"
source_dataset="CIFAR10"
target_dataset="CIFAR10-C"
subset="none"
level=5
ipc=1
sh_file="run.sh"
data_path="../data"
save_path="./results"
synset_path="./trained_synset"

num_eval=5
epoch_eval_train=1000
batch_train=256

FLAG="${target_dataset}_level${level}#${subset}#${method}_ipc${ipc}"
CUDA_VISIBLE_DEVICES=${cuda_id} python3.8 main.py \
--source_dataset ${source_dataset} --target_dataset ${target_dataset} --subset ${subset} \
--level ${level} \
--ipc ${ipc} \
--sh_file ${sh_file} \
--data_path ${data_path} --save_path ${save_path} --synset_path ${synset_path} \
--num_eval ${num_eval} --epoch_eval_train ${epoch_eval_train} --batch_train ${batch_train} \
--FLAG ${FLAG}
