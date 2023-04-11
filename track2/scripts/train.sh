export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s
# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s

config=configs/vitbase_retrieval.py

#python3 -m paddle.distributed.launch --log_dir=./logs/vitbase_retrieval --gpus="0,1,2,3,4,5,6,7"  tools/ufo_train.py --config-file ${config} #--resume 
/root/paddlejob/workspace/env_run/anaconda3/envs/py37_meta_pd-2.3.0_cu11/bin/python3.7 \
-m paddle.distributed.launch --log_dir=./logs/vitbase_retrieval --gpus="0,1,2,3,4,5,6,7"  tools/ufo_train.py --config-file ${config} #--resume

