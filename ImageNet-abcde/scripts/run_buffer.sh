cd ..
cuda_id=3
dst="imagenet-a"
res=128
net="ConvNet"
depth=5
norm_train="instancenorm"
data_path="../../../../../../data/IMAGENET2012"
buffer_path="../buffers"

train_epochs=50
num_experts=100

CUDA_VISIBLE_DEVICES=${cuda_id} python3.8 buffer.py \
--dataset=${dst} --res=${res} \
--model=${net} --depth=${depth} --norm_train=${norm_train} \
--data_path=${data_path} --buffer_path=${buffer_path} \
--train_epochs=${train_epochs} --Iteration=${num_experts}