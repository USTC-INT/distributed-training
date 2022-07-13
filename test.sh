WORKER_NUM=$1
MODEL=$2

sudo python3 src/launch.py --master 1 --ip 172.16.210.2 --worker_num $WORKER_NUM --config_file config/workers.json --dataset CIFAR100 --model $MODEL