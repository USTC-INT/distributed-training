# Distributed Training with PS Architecture

This repo is the code of distributed training (DT), where one DT task contains one PS and several workers, communicating with each other by sockets.

## Requirements

1. Run `pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/` to install python deps.
2. Run `sudo apt-get install libtiff5-dev libjpeg8-dev zlib1g-dev` to install deps for `pillow`.
3. _(Optional, we have installed cuda-version torch in most of machines)_ Run `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu -i https://mirrors.aliyun.com/pypi/simple/` to install _cpu-only_ version torch.

_If you get an error installing paramiko, which is used to create workers in other machines, you will need to install its dependencies manually._

```shell

# Update pip
python3 -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/

# Install paramiko
pip3 install paramiko -i https://pypi.mirrors.ustc.edu.cn/simple/
```

## Usage

1. Run `./deploy.sh` to sync codes among all the machines: make sure you have created the `<repo>` directory.
2. Run `./test.sh $WORKER_NUM` to start training. The scripts will run `python3 launch.py --master True xxx` to launch the PS, which will launch workers via ssh according to the IP list in config/.
