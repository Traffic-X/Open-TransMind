export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s
# kill -9 $(lsof -t /dev/nvidia*)
# sleep 1s

config=configs/test_vitbase_jointtraining_config.py

python3 -m paddle.distributed.launch --log_dir=./logs/vitbase_jointraining --gpus="0,1,2,3,4,5,6,7"  tools/ufo_test.py --config-file ${config} --eval-only 


