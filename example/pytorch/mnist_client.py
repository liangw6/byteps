import socket
import pickle
import time             # for recording performance
import multiprocessing as mp
from torchvision import datasets, transforms
import torch
import sys

import inference_utils

DATA_PATH = './MNIST/'
WORKERS = ['localhost']
PORT = 50011

BATCH_SIZE = 16 #64
REQUEST_PER_CLIENT = 10 # 100

# toy
BATCH_SIZE = 8
REQUEST_PER_CLIENT = 5

# transofrm applied by byteps train_mnist_byteps.py
DATA_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def client_send_and_recv(ip, port, message):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port))
    try:
        inference_utils.send_msg(sock, message, use_pickle=True)
        # sock.sendall(message)

        # time and get response
        after_send = time.time()
        # response = sock.recv(1024)
        response = inference_utils.recv_msg(sock, use_pickle=True)
        latency = time.time() - after_send
    finally:
        sock.close()
    return response, latency

def one_client(client_id):
    # load in data
    test_dataset = \
    datasets.MNIST(DATA_PATH, train=False, transform=DATA_TRANSFORM,
                   download=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # keep track of result
    request_cnt = 0
    accu = []
    latency = [] 
    for data, target in test_loader:
        if request_cnt >= REQUEST_PER_CLIENT:
            # only  do so many requests
            break

        worker_latency = torch.zeros(len(WORKERS))
        worker_answers = torch.zeros([len(WORKERS), BATCH_SIZE])
        for index, one_worker in enumerate(WORKERS):
            recv_data, process_latency = client_send_and_recv(one_worker, PORT, data)
            if recv_data is None:
                print("received None from worker", index)
                continue
            assert(recv_data.shape == (BATCH_SIZE, ))

            worker_answers[index, :] = recv_data
            worker_latency[index] = process_latency

        # ASSUME: concensus is the one with most vote from all workers
        worker_concensus = worker_answers.mode(0, keepdim=True)[0]
        # print("step 1")
        # ONLY FOR MNIST DATASET!!!!! converting to int avoids floating point comparison problem
        curr_accu = worker_concensus.int().eq(target.int()).sum().float() / target.numel()
        print("accuracy", curr_accu)
        curr_latency = worker_latency.sum().float() / worker_latency.numel()
        print("latency", curr_latency)

        sys.stdout.flush()
        accu.append(curr_accu)
        latency.append(curr_latency)
        request_cnt += 1

    print("Client", client_id, "cleaning up")

    # return
    return accu, latency

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python mnist_client.py n_clients")
        sys.exit()
    n_clients = int(sys.argv[1])

    pool = mp.Pool(processes=n_clients) 
    results = pool.map(one_client, range(n_clients))

    print("final results", results)
    





