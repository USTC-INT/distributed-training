# Distributed Layer Training for PS Architecture

This repo is the training code of In-network Layer Aggregation for Distributed Training. One distributed training task contains one PS and several workers, communicating with each other by sockets.

## Requirements

1. Run `pip3 install -r requirements.txt` to install python deps.
2. Run `sudo apt-get install libtiff5-dev libjpeg8-dev zlib1g-dev` to install deps for `pillow`.
3. Run `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu -i https://mirrors.aliyun.com/pypi/simple/` to install cpu-only version torch.

## Usage

Run `python3 launch.py --master True xxx` to launch the PS. The PS will launch workers via ssh according to the IP list in config/.
