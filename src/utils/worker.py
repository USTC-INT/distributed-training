import os
import pickle
import socket
import struct
import time
import paramiko
from time import sleep
from threading import Thread


work_dir = '/home/sdn/fj/distributed-layer-INA'

def killport(port):
    command = '''kill -9 $(netstat -nlp | grep :''' + str(
        port) + ''' | awk '{print $7}' | awk -F"/" '{ print $1 }')'''
    os.system(command)
    
def bind_port(listen_ip, listen_port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    start_time = time.time()
    while True:
        try:
            s.bind((listen_ip, listen_port))
            break
        except OSError as e:
            print("**OSError**", listen_ip, listen_port)
            sleep(0.7)
            killport(listen_port)
            if time.time() - start_time > 30:
                exit(1)
    s.listen()
    conn, _ = s.accept()
    return conn

def send_data(s, data):
    ser_data = pickle.dumps(data)
    s.sendall(struct.pack(">I", len(ser_data)))
    s.sendall(ser_data)

def get_data(s):
    data_len = struct.unpack(">I", s.recv(4))[0]
    data = s.recv(data_len, socket.MSG_WAITALL)
    recv_data = pickle.loads(data)
    return recv_data

class Worker:
    def __init__(self, idx, dataset, model, use_cuda,epoch, batch_size, ip, ssh_port, ps_ip, ps_port):
        self.idx = idx
        self.dataset = dataset
        self.model = model
        self.use_cuda = use_cuda
        self.epoch = epoch
        self.batch_size = batch_size
        self.worker_time = 0.0
        self.ip=ip
        self.ssh_port= ssh_port
        self.ps_ip=ps_ip
        self.ps_port=ps_port
        self.socket = None
        self.updated_paras=None
        
    def launch(self, para, partition):
        try:
            if self.ip =="127.0.0.1":
                t= Thread(target=self._launch_local_process)
                t.start()
            else:
                t= Thread(target=self._launch_remote_process)
                t.start()
        except Exception as e:
            print(e)
            exit(1)
        else:
            self._init_send_socket()
            init_config={
                'para':para,
                'train_data_index' : partition[0].use(self.idx),
                'test_data_index' : partition[1].use(self.idx)
            }
            self.send_data(init_config)

    def send_data(self, data):
        ser_data = pickle.dumps(data)
        self.socket.sendall(struct.pack(">I", len(ser_data)))
        self.socket.sendall(ser_data)
    
    def get_trained_model(self):
        try:
            data_len = struct.unpack(">I", self.socket.recv(4))[0]
            data = self.socket.recv(data_len, socket.MSG_WAITALL)
        except Exception as e:
            print(e)
            exit(1)
        else:
            self.updated_paras = pickle.loads(data)
            # self.updated_paras.to()

    def _init_send_socket(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while self.socket.connect_ex((self.ip, int(self.ps_port))) != 0:
            sleep(0.5)

    def _launch_remote_process(self):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            ssh.connect(hostname=self.ip, port=int(self.ssh_port),
                        username='sdn', password='sdn123456')
        except Exception as e:
            print(e)
            ssh.close()
            raise e
        else:
            cmd = ' cd ' + work_dir + '; sudo python3 ' + '-u src/launch.py' + \
                  ' --master ' + str(0) + \
                  ' --master_ip ' + str(self.ps_ip) + \
                  ' --master_port ' + str(self.ps_port) + \
                  ' --ip ' + str(self.ip) + \
                  ' --idx ' + str(self.idx) + \
                  ' --dataset ' + str(self.dataset) + \
                  ' --model ' + str(self.model) + \
                  ' --epoch ' + str(self.epoch) + \
                  ' --batch_size ' + str(self.batch_size) + \
                  ' > data/log/worker_' + str(self.idx) + '.txt 2>&1'
            print("Execute {}.".format(cmd))
            stdin, stdout, stderr = ssh.exec_command(cmd, get_pty=True)
            stdin.write('sdn123456' + '\n')
            output = []
            out = stdout.read()
            error = stderr.read()
            if out:
                print('[%s] OUT:\n%s' % (self.ip, out.decode('utf8')))
                output.append(out.decode('utf-8'))
                print(output)
            if error:
                print('ERROR:[%s]\n%s' % (self.ip, error.decode('utf8')))
                output.append(str(self.ip) + ' ' + error.decode('utf-8'))
                print(output)
                raise Exception("Launch Error.")
    
    def _launch_local_process(self):
        raise NotImplementedError

    # def get_config(self):
    #     self.train_info = get_data_socket(self.socket)
