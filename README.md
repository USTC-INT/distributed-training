# Distributed Layer Training for PS Architecture

This repo is the training code of In-network Layer Aggregation for Distributed Training. One distributed training task contains one PS and several workers, communicating with each other by sockets.

## Usage

Run `python3 launch.py --master True xxx` to launch the PS. The PS will launch workers via ssh according to the IP list in config/.
