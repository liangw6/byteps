import socket
import pickle
import time             # for recording performance
import multiprocessing as mp
from torchvision import datasets, transforms
import torch

import inference_utils

DATA_PATH = './MNIST/'
WORKERS = ['localhost']
PORT = 50007

BATCH_SIZE = 64
REQUEST_PER_CLIENT = 100
# transofrm applied by byteps train_mnist_byteps.py
DATA_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def one_client(client_id):
    # 1. connect to all workers
    worker_sockets = []
    for worker_ip in WORKERS:
        #             IPV4                  TCP
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))
        worker_sockets.append(s)

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
            brreak
        
        before_send = time.time()
        # send the data to workers
        for s in worker_sockets:
            inference_utils.send_msg(s, data, use_pickle=True)
            # s.send(pickle.dumps(data))

        # wait for answer from workers
        worker_answers = torch.zeros([len(worker_sockets), BATCH_SIZE])
        for index, s in enumerate(worker_sockets):
            # s.recv will block until data is received
            # WARNING: this may not be fair when there are a lot of workers
            recv_data = inference_utils.recv_msg(s, use_pickle=True)
            # recv_data = pickle.loads(s.recv(4096))
            assert(recv_data.shape == (BATCH_SIZE, ))

            worker_answers[index, :] = recv_data

        after_receive = time.time()

        # ASSUME: concensus is the one with most vote from all workers
        worker_concensus = worker_answers.mode(0, keepdim=True)[0]
        curr_accu = worker_concensus.eq(target).sum().float() / target.numel()


        # TODO: compare worker answers with target
        print("worker answers:" worker_answers)
        print("target:", target)
        accu.append(curr_accu)
        latency.append(after_receive - before_send)
        tot_requests += 1

    print("Client", client_id, "cleaning up")

    # clean up
    for s in worker_sockets:
        s.close()
    # return
    return accu, latency

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python mnist_client.py n_clients")
        sys.exit()
    n_clients = int(sys.argv[1])

    with mp.pool as pool:
        results = pool.map(one_client, range(n_clients))

    print("final results", results)
    





