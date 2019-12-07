# Development

## Dev Questions

1. How does multiple-GPU work? Is that already handled by torch?

2. Different workers may have different predictions... What should we do?

3. Byteps_push_pull seems to do all the aggregate for us. My guess is that they are also using the server

4. What codes do server / scheduler run anyways?

## Dev Design & Reasoning

* Each client needs to talk all workers. This overhead cannot be avoided because the ML model is sharded throughout all workers, and a response HAS TO be the aggregate for the all workers. The only solution would be to do replciation of weights, which can be a potentially interesting topic

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
