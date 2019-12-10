# Development

## Dev Questions

1. How does multiple-GPU work? Is that already handled by torch?

2. Different workers may have different predictions... What should we do?

3. Byteps_push_pull seems to do all the aggregate for us. My guess is that they are also using the server

4. What codes do server / scheduler run anyways?

## Dev Design & Reasoning

* Each client needs to talk all workers. This overhead cannot be avoided because the ML model is sharded throughout all workers, and a response HAS TO be the aggregate for the all workers. The only solution would be to do replciation of weights, which can be a potentially interesting topic

* Without Replication, each worker serves the funcion of both training and real-time inferencing. This will result in resource competition between both tasks, but on the flip side, it adds real-time inferencing without additional machines, thus useful in certain situations with limited compute resources.

* This is more like a concern or question. I thought byteps separates a model in multiple worker machines. But mnist_train gathers test accuracy through aggregating and averaging test_loss from all workers. So that is why I decided to gather predictions the same way. But if a model is truly separated between different worker machines, this may not get the most accurate prediction. (On the other hand though, how do you gather prediction in this kind of system...)

## Dev Progress Update

* Almost working real-time inference system on mnist example scripts

* Current setup:
    * example/pytorch/mnist_client.py: a psudo-client script that simulates a remote client sending requests from the web.

        * In this case, client request consists of a batch of images, or precisely pytorch tensors, to be classified to 0-9 numbers

        * a client script supports multi-processing, simulating multiple clients running requests separately. We can also run the same script on many machines, simulating even larger groups of clients

        * a client may also initiate many requests. In current development, a client will use a pytorch loader to load mnist test dataset, and send batches of test images, through TCP socket, to worker machines.

        * a client needs to send the same image to all worker machines. It will then receive response from all worker machines and choose the most common worker predictions as "best" worker answer. 

        * For each request, a client records both accuracy and latency, which can be used for analysis later

    * example/pytorch/mnist_worker.py: a psudo-worker script that simulates a worker with  both training and real-time inference

        * The worker essentially has two threads with a globally shared variable `model`. One thread runs training with gradient descent, i.e. `writing` to the model, while the other runs predictions, i.e. `read only` to the model

        * The training thread is exactly the same as byteps.

        * The prediction thread is basically a server. It waits for clients to connect,spawns a new thread for each newly connected client and continues in waiting.

        * the newly spawned thread simply runs prediction for the model, the same way as test() method in mnist_train does. Then it sends the prediction to the client.

        * Since `model` variable is a global variable, once it gets updated in training thread, the updated value will show up for prediction threads. This is demonstrated by a toy example in inference_utils/experiments.ipynb.
        
        * since updating loss for a model may not be atomic, the prediction thread may see intermediate weights that are being updated. We can avoid this easily with a lock or something, but I just want to see if we can get away with it.

* Current Progress: The client can initiate one request to a worker and gets a response back. It is also able to use that reponse to calculate accuracy and latency. To demo this, you will need two containers

In the container for worker, 
```bash
# basically run it the same way as before
# ... omitting all environment variables setup
python /usr/local/byteps/launcher/launch.py   \
    /usr/local/byteps/example/pytorch/start_pytorch_byteps.sh --epochs 1

# you will then see the following prints
BytePS launching worker
training mnist... with real-time inference
...
starts real-time inference server
```
in byteps-client container
```bash
# 1 for just one client
python example/pytorch/mnist_client.py mnist_tmp.pkl --nclient 24
# you will then see the following prints
('Client', 4, 'cleaning up')
('Client', 3, 'cleaning up')
('Client', 22, 'cleaning up')
('Client', 20, 'cleaning up')
('Client', 1, 'cleaning up')
('Client', 12, 'cleaning up')
('Client', 13, 'cleaning up')
('Client', 16, 'cleaning up')
('Client', 6, 'cleaning up')
('Client', 10, 'cleaning up')
('Client', 21, 'cleaning up')
('Client', 14, 'cleaning up')
('Client', 23, 'cleaning up')
('Client', 9, 'cleaning up')
('Client', 8, 'cleaning up')
('Client', 7, 'cleaning up')
('Client', 11, 'cleaning up')
('Client', 0, 'cleaning up')
('Client', 17, 'cleaning up')
('Client', 18, 'cleaning up')
('Client', 19, 'cleaning up')
('Client', 2, 'cleaning up')
('Client', 5, 'cleaning up')
('Client', 15, 'cleaning up')
received final result
# final result will be stored in pickle file mnist_tmp.pkl
```

* Curernt Problem: if the client requests 30 or more concurrent connections, it would get connection reset by peer error. Stackoverflow says it means the server crashed, but I am not sure what causes the crash. Maybe it is just my computer, since I am running both server and client on the same machine. TODO: try on AWS machines 
```
Traceback (most recent call last):
  File "example/pytorch/mnist_client.py", line 96, in <module>
    results = pool.map(one_client, range(args.nclient))
  File "/usr/lib/python2.7/multiprocessing/pool.py", line 251, in map
    return self.map_async(func, iterable, chunksize).get()
  File "/usr/lib/python2.7/multiprocessing/pool.py", line 567, in get
    raise self._value
socket.error: [Errno 104] Connection reset by peer
```

* Other TODOS:

    * Analyze the results

        * How much does training get impaired by real-time inference? training time & accuracy w/ vs. wo/ real-time server

        * How much latency / inaccuracy was introduced?

            * test on with vs. without network latency?

    * Try harder networks? e.g. resnet?



## Dev Environment

### About docker
```bash
# If running below command, and you accidentally ctrl-D exit the program
# The container will be stopped
# docker run -it  ...

# Instead, run docker detached, so it will continiue running
# give it a name such as byteps-dev
docker run -dit --name byteps-dev --net=host --gpus all --shm-size=32768m bytepsimage/worker_pytorch
# then attach to it
docker attach byteps-dev

# now docker will always be there even after you accidently exited it
```

### Vesion Control

Develop inside pre-built docker image, and then somehow migrate the changes to this repo

```bash
# use pre-built docker image from byteps as before
nvidia-docker run -it --net=host --shm-size=32768m bytepsimage/pytorch bash

# [optional] set byteps repo as upstream
git remote add upstream origin

# Soln 1: replace existing with ours...
git remote set-url origin  https://github.com/liangw6/byteps.git
```

## Dev Goal

Adds real-time inference capability to existing byteps

## Dev Plan

### Plan 1. Use the most up-to-date weights on existing worker machines

* On existing worker machines, implement new threads / processes that will continuously listen to network for real-time inference command

* Upon receiving any, threads send this request through workers in the same way as test()

* After getting from worker, threads returned to users

Technical Detail

```Python
# Use byteps push_pull to get predictions from all workers
import byteps.torch as bps
def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = bps.push_pull(tensor, name=name)
    return avg_tensor.item()

# TODO: Different workers may have different predictions... What should we do?


```

`Discussion`:

* Cleanest implementation and reuses most resources and codes

* May use non-consistent weights, although inconsistency is only between workers.

`TODO`
* Evaluate the inconsistency between workers: How different can they be? How much synchronization is there already

* Can same GPU handle training, e.g. loss.backward(), while an inference request comes in?
```
Experiemnts to consider (Consider recording data for Accuracy & Latency):
1. For each iteration in each epoch, sequentially first execute any outstanding inference requests
2. Have a different thread, with the same model (maybe periodically updated model), thread always goes ahead and execute the real-time inference request
3. Have a different machine / processs, with a replicated model (periodically updated), which would always wait and execute real-time inference request
```


### Plan 2. Use Replicated Best Weights from New Worker Machines

* Run two instances of Byteps on two set of machines, one for training and the other for real-tiem inference

* Periodically, training worker machiens send updated weights to real-time inference machines

* Real-time inference group of machines:

  * Worker machines need to have a process to constantly listen to incoming user command

  * Worker machines need to have a process / thread to constantly listen to new weight updates and enforce them, similar to local_reduce
