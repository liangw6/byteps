# Development

## Dev Questions

1. How does multiple-GPU work? Is that already handled by torch?

2. Different workers may have different predictions... What should we do?

3. Byteps_push_pull seems to do all the aggregate for us. My guess is that they are also using the server

4. What codes do server / scheduler run anyways?

## Dev Environment

1. Develop inside custom-built docker image for this repo

```bash
# the following commands have NOT been tested

# builds a docker image called fan_club
docker build -t fan_club -f docker/Dockerfile.fanclub_pytorch .

nvidia-docker run -it fan_club bash
```

2. Develop inside pre-built docker image, and then somehow migrate the changes to this repo

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

### Plan 2. Use Replicated Best Weights from New Worker Machines

* Run two instances of Byteps on two set of machines, one for training and the other for real-tiem inference

* Periodically, training worker machiens send updated weights to real-time inference machines

* Real-time inference group of machines:

  * Worker machines need to have a process to constantly listen to incoming user command

  * Worker machines need to have a process / thread to constantly listen to new weight updates and enforce them, similar to local_reduce
