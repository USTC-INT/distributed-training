# Distributed Training with PS Architecture

This repo is the training code of In-network Layer Aggregation for Distributed Training. One distributed training task contains one PS and several workers, communicating with each other by sockets.

## Requirements

### Install setuptools-rust

```
export RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
export RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip3 install setuptools-rust -i https://mirrors.aliyun.com/pypi/simple/
```

### Install paramiko
`python3 -m pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/`

1. Run `pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/` to install python deps.
2. Run `sudo apt-get install libtiff5-dev libjpeg8-dev zlib1g-dev` to install deps for `pillow`.
3. Run `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu -i https://mirrors.aliyun.com/pypi/simple/` to install cpu-only version torch.


## Usage

Run `python3 launch.py --master True xxx` to launch the PS. The PS will launch workers via ssh according to the IP list in config/.
