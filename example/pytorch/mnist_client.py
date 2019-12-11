import socket
import pickle
import time             # for recording performance
import multiprocessing as mp
from torchvision import datasets, transforms
import torch
import sys
import argparse

import inference_utils
DATA_PATH = './MNIST/'
WORKERS = ['localhost']
PORT = 50015

parser = argparse.ArgumentParser(description='PyTorch MNIST real-time inference client example')
parser.add_argument('output_file', metavar='output.pkl', help='pickle file name to save the final result')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--request-per-client', type=int, default=1000, metavar='N',
                    help='number of requests for each client process (default: 1000)')
parser.add_argument('--nclient', type=int, default=8, metavar='N',
                    help='number of clients in total, each of which run on separate process (default: 8)')
args = parser.parse_args()

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
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)
    
    # keep track of result
    request_cnt = 0
    accu = []
    latency = []
    while True:
        for data, target in test_loader:
            if request_cnt >= args.request_per_client:
                # only  do so many requests
                print("Client", client_id, "cleaning up")
                # return
                return accu, latency
            worker_latency = torch.zeros(len(WORKERS))
            worker_answers = torch.zeros([len(WORKERS), args.batch_size])
            for index, one_worker in enumerate(WORKERS):
                recv_data, process_latency = client_send_and_recv(one_worker, PORT, data)
                if recv_data is None:
                    print("received None from worker", index)
                    continue
                assert(recv_data.shape == (args.batch_size, ))

                worker_answers[index, :] = recv_data
                worker_latency[index] = process_latency

            # ASSUME: concensus is the one with most vote from all workers
            worker_concensus = worker_answers.mode(0, keepdim=True)[0]
            # print("step 1")
            # ONLY FOR MNIST DATASET!!!!! converting to int avoids floating point comparison problem
            curr_accu = worker_concensus.int().eq(target.int()).sum().float() / target.numel()
            # print("accuracy", curr_accu)
            curr_latency = worker_latency.sum().float() / worker_latency.numel()
            # print("latency", curr_latency)

            sys.stdout.flush()
            accu.append(curr_accu.item())
            latency.append(curr_latency.item())
            request_cnt += 1
        # should automatically loop through test_loader again and again
        print("finished one pass through test, request_cnt", request_cnt)

if __name__ == "__main__":
    pool = mp.Pool(processes=args.nclient)
    results = pool.map(one_client, range(args.nclient))

    print("received final result")

    with open(args.output_file, 'w') as output_file:
        pickle.dump(results, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    





