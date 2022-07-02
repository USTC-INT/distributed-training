import argparse
import asyncio
import concurrent.futures
import json
import math
import random
import time
import numpy as np
import torch
import os

from utils import dataset
from utils import models
from utils import worker


parser = argparse.ArgumentParser(description='Distributed Model Training')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--master', type=bool, default=True)
parser.add_argument('--master_ip', type=str, default='127.0.0.1')
parser.add_argument('--master_port', type=int, default=53300)
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=int, default=53300)
parser.add_argument('--worker_num', type=int, default=8)
parser.add_argument('--config_file', type = str, default="./config.json")
parser.add_argument('--model', type=str, default='alexnet')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=100)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

master_listen_port_base = args.port + random.randint(0,20) * 20

def master_loop():
    worker_list=[]
    step_size=1

    try:
        global_model = models.get_model(args.model)
    except TypeError as t:
        print(t)
        exit(1)
    else:
        init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
        print("Model name: {}".format(args.model))
        print("Model Size: {} MB".format(init_para.nelement() * 4 / 1024 / 1024))

    try:
        _, test_dataset = dataset.load_datasets(args.dataset)
    except TypeError as t:
        print(t)
        exit(1)
    else:
        test_loader = dataset.create_dataloaders(test_dataset, batch_size=args.batch_size, shuffle=False)
        train_data_partition, test_data_partition = dataset.partition_data(args.dataset, args.worker_num)
    
    try:
        file = open(args.config_file)
    except FileNotFoundError as e:
        print(e)
        exit(1)
    else:
        host_config=json.load(file)
        for i in range(args.worker_num):
            worker_list.append(
                worker.Worker(
                    i, 
                    args.dataset, 
                    args.model, 
                    True, 
                    args.epoch, 
                    args.batch_size, 
                    host_config[i]['host_ip'], 
                    host_config[i]['ssh_port'],
                    args.ip, 
                    master_listen_port_base+i
                )
            )
    finally:
        file.close()

    try:
        print("Launch workers and send init paras.")
        communication_parallel(worker_list, action="init", para= init_para, partition=(train_data_partition, test_data_partition))
    except Exception as e:
        print(e)
        for w in worker_list:
            w.socket.shutdown(2)
        exit(1)
    else:
        print("Start Training...")

        global_model.to(device)
        for epoch_idx in range(args.epoch):
            start_time=time.time()
            communication_parallel(worker_list, action="get_model")
            global_para = torch.nn.utils.parameters_to_vector(global_model.parameters()).clone().detach()
            global_para = aggregate(global_para, worker_list, step_size)
            
            communication_parallel(worker_list, action="send_model", data=global_para)
            torch.nn.utils.vector_to_parameters(global_para, global_model.parameters())
            loss, acc = test(global_model, test_loader, device)
            duration = time.time() - start_time
            print("Epoch: {}, accuracy = {}, loss = {}, duration = {}\n".format(epoch_idx, acc, loss, duration))
    finally:
        for w in worker_list:
            w.socket.shutdown(2)

def worker_loop():
    lr=0.01
    min_lr=0.001
    decay_rate=0.97
    weight_decay=0.0
    ratio=0.2

    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    
    master_socket = worker.bind_port(args.ip, args.master_port)
    config = worker.get_data(master_socket)

    print('Create local model.')

    local_model = models.get_model(args.model)
    torch.nn.utils.vector_to_parameters(config['para'], local_model.parameters())
    local_model.to(device)
    train_dataset, test_dataset = dataset.load_datasets(args.dataset)
    train_loader = dataset.create_dataloaders(train_dataset, batch_size=args.batch_size,
                                               selected_idxs=config["train_data_idxes"])
    test_loader = dataset.create_dataloaders(test_dataset, batch_size=args.batch_size, shuffle=False)

    local_model.to(device)
    local_steps = 50

    for epoch in range(args.epoch):
        epoch_lr = max((decay_rate * lr, min_lr))
        print("model-{}-epoch-{} ".
              format(args.model, epoch))
        optimizer = torch.optim.SGD(local_model.parameters(), lr=epoch_lr, weight_decay=weight_decay)
        train_loss = train(local_model, train_loader, optimizer, local_iters=local_steps, device=device)
        local_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).clone().detach()

        test_loss, acc = test(local_model, test_loader, device)
        print("after aggregation, epoch: {}, train loss: {}, test loss: {}, test accuracy: {}".format(epoch, train_loss,
                                                                                                      test_loss, acc))
        
        worker.send_data(master_socket, local_para)

        local_para = worker.get_data(master_socket)
        local_para.to(device)
        torch.nn.utils.vector_to_parameters(local_para, local_model.parameters())

    master_socket.shutdown(2)
    master_socket.close()

def aggregate(local_para, worker_list, step_size):
    with torch.no_grad():
        para_delta = torch.zeros_like(local_para)
        average_weight = 1.0 / (len(worker_list) + 1)
        for worker in worker_list:
            model_delta = worker.config.neighbor_paras - local_para
            para_delta += average_weight * step_size * model_delta

        local_para += para_delta

    return local_para

def test(model, data_loader, device):
    model.eval()
    data_loader = data_loader.loader
    loss = 0.0
    accuracy = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
            loss += loss_func(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(1, keepdim=True)
            batch_correct = pred.eq(target.view_as(pred)).sum().item()
            correct += batch_correct

    loss /= len(data_loader.dataset)
    accuracy = np.float(1.0 * correct / len(data_loader.dataset))

    return loss, accuracy

def train(model, data_loader, optimizer, local_iters=None, device=None):
    model.train()
    if local_iters is None:
        local_iters = math.ceil(len(data_loader.loader.dataset) / data_loader.loader.batch_size)
    
    train_loss = 0.0
    samples_num = 0
    for iter_idx in range(local_iters):
        data, target = next(data_loader)
        data, target = data.to(device), target.to(device)
        output = model(data)
        optimizer.zero_grad()
        loss_func=torch.nn.CrossEntropyLoss()
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        train_loss += (loss.item() * data.size(0))
        samples_num += data.size(0)

    if samples_num != 0:
        train_loss /= samples_num

    return train_loss

def communication_parallel(worker_list, action, para=None,data=None, partition=None):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_list), )
        tasks = []
        for worker in worker_list:
            if action == "init":
                tasks.append(loop.run_in_executor(executor, worker.launch, para, partition))
            elif action == "get_model":
                tasks.append(loop.run_in_executor(executor, get_compressed_model_top, worker.config, worker.socket,
                                                  worker.para_nums))
            elif action == "send_model":
                tasks.append(loop.run_in_executor(executor, worker.send_data, data))
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
    except:
        exit(1)

if __name__ == "__main__":
    if args.master:
        master_loop()
    else:
        worker_loop()
