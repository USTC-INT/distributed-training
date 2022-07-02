import argparse
import asyncio
import concurrent.futures
import json
import random
import numpy as np
import torch
import os

parser = argparse.ArgumentParser(description='Parameter Server')
parser.add_argument('--model', type=str, default='alexnet')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--worker_num', type=int, default=8)
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--nic_ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=100)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

def main():
    lr = 0.01

    global_model = models.get_model(args.model)
    init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())

    para_nums = torch.nn.utils.parameters_to_vector(global_model.parameters()).nelement()
    model_size = init_para.nelement() * 4 / 1024 / 1024
    print("Model name: {}".format(args.model))
    print("Model Size: {} MB".format(model_size))

    nic_socket = socket.socket(socket.AF_INET, socket.SOCK_RAW, NGA_TYPE)
    nic_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 20480000)
    print("Recv buff: {}".format(nic_socket.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)))
    nic_socket.bind((args.nic_ip, 0))
    updated_para = []
    recv_thread = RecvThread(func=get_data_from_nic, args=(nic_socket, updated_para, args.nic_ip))
    recv_thread.start()

    train_data_partition, test_data_partition = partition_data(args.dataset, args.data_pattern, args.worker_num)
    
    for worker_idx, worker in enumerate(worker_list):
        worker.config.para = init_para
        worker.config.custom["train_data_idxes"] = train_data_partition.use(worker_idx)
        worker.config.custom["test_data_idxes"] = test_data_partition.use(worker_idx)
    print("Try to connect socket and send init config.")
    try:
        communication_parallel(worker_list, action="init")
    except Exception as e:
        for worker in worker_list:
            worker.socket.shutdown(2)
        return
    else:
        print("SUCCESSFUL: inition done.")

    global_model.to(device)
    train_dataset, test_dataset = datasets.load_datasets(args.dataset)
    test_loader = datasets.create_dataloaders(test_dataset, batch_size=args.batch_size, shuffle=False)

    total_time = 0.0
    epoch_time = []
    for epoch_idx in range(args.epoch):
        start_time = time.time()
        print("get begin")
        communication_parallel(worker_list, action="get_model")
        print("get end")
        global_para = torch.nn.utils.parameters_to_vector(global_model.parameters()).clone().detach()
        aggregate_model_from_nic(global_para, updated_para, args.step_size, worker_num)
        updated_para.clear()
        global_para = aggregate_model(global_para, worker_list, args.step_size)

        print("send begin")
        communication_parallel(worker_list, action="send_model", data=global_para)
        # communication_parallel(worker_list, action="send_model", data=tmp_para)
        print("send end")

        torch.nn.utils.vector_to_parameters(global_para, global_model.parameters())
        test_loss, acc = test(global_model, test_loader, device, model_type=args.model)
        end_time = time.time() - start_time
        epoch_time.append(end_time)
        print("Epoch: {}, accuracy: {}, test_loss: {}\n".format(epoch_idx, acc, test_loss))
        print("epoch time: {}".format(str(end_time)))

    for worker in worker_list:
        worker.socket.shutdown(2)


def aggregate_model(local_para, worker_list, step_size):
    with torch.no_grad():
        para_delta = torch.zeros_like(local_para)
        average_weight = 1.0 / (len(worker_list) + 1)
        for worker in worker_list:
            model_delta = worker.config.neighbor_paras - local_para
            para_delta += average_weight * step_size * model_delta

        local_para += para_delta

    return local_para


def get_nic_data(recv_data, length):
    sequence_payload = [0.0 for i in range(length)]
    count = 0
    for raw_data in recv_data:
        nga_header = NGAHeader(raw_data[:HEADER_BYTE])
        nga_payload = NGAPayload(raw_data[HEADER_BYTE - 1:])
        count += len(nga_payload.data)
        for i in range(DATA_NUM):
            if nga_header.sequenceid * DATA_NUM + i >= length:
                break
            sequence_payload[nga_header.sequenceid * DATA_NUM + i] += nga_payload.data[i]
    # print("Len of data from nic: {}.".format(count))
    return sequence_payload


def aggregate_model_from_nic(local_para, recv_data, step_size, worker_num):
    payload = get_nic_data(recv_data, len(local_para))
    # updated_para = torch.Tensor(payload)
    # average_weight = 1.0 / (worker_num + 1)
    # with torch.no_grad():
    #     delta = (local_para - updated_para)
    #     local_para += (step_size * delta * average_weight)

    return local_para


def communication_parallel(worker_list, action, data=None):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_list), )
        tasks = []
        for worker in worker_list:
            if action == "init":
                tasks.append(loop.run_in_executor(executor, worker.send_init_config))
            elif action == "get_model":
                tasks.append(loop.run_in_executor(executor, get_compressed_model_top, worker.config, worker.socket,
                                                  worker.para_nums))
            elif action == "send_model":
                tasks.append(loop.run_in_executor(executor, worker.send_data, data))
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
    except:
        sys.exit(0)


def get_compressed_model_top(config, socket, nelement):
    try:
        received_para = get_data_socket(socket)
    except Exception as e:
        print("FAILED: get model error.")
        print(e)
        sys.exit(1)
    else:
        received_para.to(device)
        config.neighbor_paras = received_para


def non_iid_partition(ratio, worker_num=10):
    partition_sizes = np.ones((10, worker_num)) * ((1 - ratio) / (worker_num - 1))

    for worker_idx in range(worker_num):
        partition_sizes[worker_idx][worker_idx] = ratio

    return partition_sizes


def partition_data(dataset_type, data_pattern, worker_num):
    train_dataset, test_dataset = datasets.load_datasets(dataset_type)

    if dataset_type == "CIFAR100":
        test_partition_sizes = np.ones((100, worker_num)) * (1 / worker_num)
        partition_sizes = np.ones((100, worker_num)) * (1 / (worker_num - data_pattern))
        for worker_idx in range(worker_num):
            tmp_idx = worker_idx
            for _ in range(data_pattern):
                partition_sizes[tmp_idx * worker_num:(tmp_idx + 1) * worker_num, worker_idx] = 0
                tmp_idx = (tmp_idx + 1) % 10
    elif dataset_type == "CIFAR10":
        test_partition_sizes = np.ones((10, worker_num)) * (1 / worker_num)
        if data_pattern == 0:
            partition_sizes = np.ones((10, worker_num)) * (1.0 / worker_num)
        elif data_pattern == 1:
            partition_sizes = [
                [0.0, 0.0, 0.0, 0.1482, 0.1482, 0.1482, 0.148, 0.1482, 0.1482, 0.111],
                [0.0, 0.0, 0.0, 0.1482, 0.1482, 0.1482, 0.1482, 0.148, 0.1482, 0.111],
                [0.0, 0.0, 0.0, 0.1482, 0.1482, 0.1482, 0.1482, 0.1482, 0.148, 0.111],
                [0.148, 0.1482, 0.1482, 0.0, 0.0, 0.0, 0.1482, 0.1482, 0.1482, 0.111],
                [0.1482, 0.148, 0.1482, 0.0, 0.0, 0.0, 0.1482, 0.1482, 0.1482, 0.111],
                [0.1482, 0.1482, 0.148, 0.0, 0.0, 0.0, 0.1482, 0.1482, 0.1472, 0.112],
                [0.1482, 0.1482, 0.1482, 0.148, 0.1482, 0.1482, 0.0, 0.0, 0.0, 0.111],
                [0.1482, 0.1482, 0.1482, 0.1482, 0.148, 0.1482, 0.0, 0.0, 0.0, 0.111],
                [0.1482, 0.1482, 0.1482, 0.1482, 0.1482, 0.148, 0.0, 0.0, 0.0, 0.111],
                [0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.111, 0.112, 0.0],
            ]
        elif data_pattern == 2:
            partition_sizes = [
                [0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                [0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                [0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                [0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                [0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125],
                [0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125],
                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125],
                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125],
                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0],
                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0],
            ]
        elif data_pattern == 3:
            partition_sizes = [[0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0, 0.0, 0.0],
                               [0.0, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0, 0.0],
                               [0.0, 0.0, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0],
                               [0.0, 0.0, 0.0, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432],
                               [0.1432, 0.0, 0.0, 0.0, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428],
                               [0.1428, 0.1432, 0.0, 0.0, 0.0, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428],
                               [0.1428, 0.1428, 0.1432, 0.0, 0.0, 0.0, 0.1428, 0.1428, 0.1428, 0.1428],
                               [0.1428, 0.1428, 0.1428, 0.1432, 0.0, 0.0, 0.0, 0.1428, 0.1428, 0.1428],
                               [0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0, 0.0, 0.0, 0.1428, 0.1428],
                               [0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432, 0.0, 0.0, 0.0, 0.1428],
                               ]
        elif data_pattern == 4:
            partition_sizes = [[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0],
                               [0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0],
                               [0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                               [0.125, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                               [0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
                               [0.125, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125, 0.125],
                               [0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125, 0.125],
                               [0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125, 0.125],
                               [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.125, 0.125],
                               [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0, 0.0, 0.125],
                               ]
        elif data_pattern == 5:
            non_iid_ratio = 0.2
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 6:
            non_iid_ratio = 0.4
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 7:
            non_iid_ratio = 0.8
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 8:
            non_iid_ratio = 0.6
            partition_sizes = non_iid_partition(non_iid_ratio)
        elif data_pattern == 9:
            non_iid_ratio = 0.9
            partition_sizes = non_iid_partition(non_iid_ratio)
        # elif data_pattern == 10:
        #     non_iid_ratio = 0.5
        #     partition_sizes = non_iid_partition(non_iid_ratio)

    train_data_partition = datasets.LabelwisePartitioner(train_dataset, partition_sizes=partition_sizes)
    test_data_partition = datasets.LabelwisePartitioner(test_dataset, partition_sizes=test_partition_sizes)

    return train_data_partition, test_data_partition

if __name__ == "__main__":
    main()
