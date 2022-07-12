import argparse
import asyncio
import concurrent.futures
import json
import math
import random
import time
from tracemalloc import start
import numpy as np
import torch
import os

from utils import datasets
from utils import models
from utils import worker

parser = argparse.ArgumentParser(description='Distributed Model Training')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--master', type=int)
parser.add_argument('--master_ip', type=str, default='127.0.0.1')
parser.add_argument('--master_port', type=int, default=53300)
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--base_port', type=int, default=53300)
parser.add_argument('--worker_num', type=int, default=8)
parser.add_argument('--config_file', type = str, default="./config.json")
parser.add_argument('--model', type=str, default='alexnet')
parser.add_argument('--dataset', type=str, default='CIFAR10')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epoch', type=int, default=100)

args = parser.parse_args()

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ip=args.ip

def aggregate(local_para, worker_list, step_size):
    with torch.no_grad():
        para_delta = torch.zeros_like(local_para)
        average_weight = 1.0 / (len(worker_list) + 1)
        for w in worker_list:
            model_delta = w.updated_paras - local_para
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

def communication_parallel(worker_list, action, para=None,updated_data=None, partition=None):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(worker_list), )
    tasks = []
    for w in worker_list:
        if action == "init":
            tasks.append(loop.run_in_executor(executor, w.launch, para, partition))
        elif action == "pull":
            tasks.append(loop.run_in_executor(executor, w.get_trained_model))
        elif action == "push":
            tasks.append(loop.run_in_executor(executor, w.send_data, updated_data))
    loop.run_until_complete(asyncio.wait(tasks))
    loop.close()

def master_loop(model, dataset, worker_num, config_file, batch_size, epoch, base_port):
    master_listen_port_base = base_port + random.randint(0,20) * 20
    worker_list=[]
    step_size=1

    try:
        global_model = models.get_model(model)
    except TypeError as t:
        print(t)
        exit(1)
        
    try:
        train_dataset, test_dataset = datasets.load_dataset(dataset,CURRENT_PATH+'/../data/datasets')
    except TypeError as t:
        print(t)
        exit(1)

    file = open(config_file)
    try:
        host_config=json.load(file)
        for i in range(worker_num):
            worker_list.append(
                worker.Worker(
                    i, 
                    dataset, 
                    model, 
                    True, 
                    epoch, 
                    batch_size, 
                    host_config[i]['host_ip'], 
                    host_config[i]['ssh_port'],
                    ip, 
                    master_listen_port_base+i
                )
            )
    except KeyError as e:
        print(e)
        exit(1)
    finally:
        file.close()

    init_para = torch.nn.utils.parameters_to_vector(global_model.parameters())
    print("Model {}: {} MB".format(model,init_para.nelement() * init_para.element_size() / 1024 / 1024))
    
    test_loader = datasets.create_dataloaders(test_dataset, batch_size,shuffle=False)
    train_data_partition, test_data_partition = datasets.partition_data(dataset, worker_num, train_dataset, test_dataset)
    
    try:
        print("Launch workers and send init paras.")
        communication_parallel(worker_list, action="init", para=init_para, partition=(train_data_partition, test_data_partition))
    except Exception as e:
        print(e)
        for w in worker_list:
            w.socket.shutdown(2)
        exit(1)
    else:
        print("Start training...")
        global_model.to(device)
        for epoch_idx in range(epoch):
            comm_time=0
            start_time=time.time()
            communication_parallel(worker_list, action="pull")
            comm_time+=(time.time()- max([w.sending_time for w in worker_list]))
            delta_time = max([w.sending_time for w in worker_list]) - min([w.sending_time for w in worker_list])

            global_para = torch.nn.utils.parameters_to_vector(global_model.parameters()).clone().detach()
            global_para = aggregate(global_para, worker_list, step_size)
            
            communication_parallel(worker_list, action="push", updated_data=global_para)

            duration = time.time() - start_time
            
            torch.nn.utils.vector_to_parameters(global_para, global_model.parameters())
            loss, acc = test(global_model, test_loader, device)

            print("Epoch: {}, accuracy = {}, loss = {}.".format(epoch_idx, acc, loss))
            print("Epoch duration => {} sec, comm => {} sec, delta => {} sec".format(duration, comm_time, delta_time))
    finally:
        for w in worker_list:
            w.socket.shutdown(2)

def worker_loop(model, dataset,idx, batch_size, epoch, master_port):
    lr=0.01
    min_lr=0.001
    decay_rate=0.97
    weight_decay=0.0
    
    master_socket = worker.bind_port(ip, master_port)
    config = worker.get_data(master_socket)
    if config == None:
        print("Worker {} :No config received.".format(idx))
        master_socket.close()
        exit(1)

    try: 
        print('Create local model {}...'.format(model))
        local_model = models.get_model(model)
    except ValueError as e:
        print(e)
        master_socket.close()
        exit(1)
    
    try:
        print('Load dataset {}...'.format(dataset))
        train_dataset, test_dataset = datasets.load_dataset(dataset,CURRENT_PATH+'/../data/datasets')
    except ValueError as e:
        print(e)
    else:
        train_loader = datasets.create_dataloaders(train_dataset, batch_size,selected_idxs=config["train_data_index"])
        test_loader = datasets.create_dataloaders(test_dataset, batch_size, shuffle=False)
        torch.nn.utils.vector_to_parameters(config['para'], local_model.parameters())
        local_model.to(device)
        local_steps = 50

        for epoch in range(epoch):
            epoch_lr = max((decay_rate * lr, min_lr))
            optimizer = torch.optim.SGD(local_model.parameters(), lr=epoch_lr, weight_decay=weight_decay)
            train_loss = train(local_model, train_loader, optimizer, local_iters=local_steps, device=device)
            local_para = torch.nn.utils.parameters_to_vector(local_model.parameters()).clone().detach()
            test_loss, acc = test(local_model, test_loader, device)
            
            print("Epoch {}: train loss = {}, test loss = {}, test accuracy = {}".format(epoch, train_loss,test_loss, acc))
            
            push_time=time.time()
            worker.send_data(master_socket, push_time, local_para)

            local_para = worker.get_data(master_socket)

            torch.nn.utils.vector_to_parameters(local_para, local_model.parameters())
    finally:
        master_socket.shutdown(2)
        master_socket.close()

if __name__ == "__main__":
    if args.master == 1:
        master_loop(args.model, args.dataset, args.worker_num, args.config_file, args.batch_size, args.epoch, args.base_port)
    else:
        worker_loop(args.model, args.dataset, args.idx, args.batch_size, args.epoch, args.master_port)
