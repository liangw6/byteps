How to set up a Amazon Spot Instance for bytePS:
1. Create your AWS account

2. create an instance on EC2 using the Create Instance Button.

3. Choose Ubuntu 16.04, choose p2.xlarge (or something with more than 1 GPU), go to the next page and check 'Request Spot Instances'.

4. Go to Add Storage and change 8 to 30. 

5. Make a key. I named it 'testing', save the private key somewhere where you can access it. 

6. You should start by sshing into something like 
   "ssh -i testing.pem ubuntu@{Public DNS (IPv4)}" 
where '{Public DNS (IPv4)}' is something like: ec2-18-233-63-34.compute-1.amazonaws.com
ec2-54-172-106-100.compute-1.amazonaws.com

7. If this doesn't work, that means that the cluster you chose was jank af and you should choose a different type (don't just take it directly from Spot (I spent 1 hour finding out that it wouldn't let you ssh into these and the online terminal also didn't work)).

8. Install docker
To install docker, follow this: 
https://medium.com/@cjus/installing-docker-ce-on-an-aws-ec2-instance-running-ubuntu-16-04-f42fe7e80869
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce
sudo usermod -aG docker ${USER}
```

9. Then you should install nvidia stuff again 
Found here https://devtalk.nvidia.com/default/topic/1065310/linux/docker-tensorflow-gpu-can-t-find-device-as-well-as-nvidia-smi-quot-no-device-found-quot-/
```
# removes nvidia stuff, if it exists
sudo apt-get remove --purge '^nvidia-.*'
sudo apt-get remove --purge '^cuda.*'
# downloads stuff correctly
sudo add-apt-repository ppa:graphics-drivers/ppa 
sudo apt update
sudo apt upgrade            # You can just keep the grub things
sudo apt install nvidia-430 # Maybe nvidia-driver-430 depending on the mood
sudo reboot                 # Gotta restart to make it work
```

10. Then follow the instructions here
https://github.com/NVIDIA/nvidia-docker
 - Apparently there is no more nvidia-docker, just use docker --gpus
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

11. Pull docker image 
Follow pytorch instructions here: https://github.com/bytedance/byteps/blob/master/docs/step-by-step-tutorial.md
 - altered below to work without nvidia-docker
```
docker pull bytepsimage/worker_pytorch
docker run -it --net=host --gpus all --shm-size=32768m bytepsimage/worker_pytorch bash

# now you are in docker environment
export NVIDIA_VISIBLE_DEVICES=0,1,2,3  # say you have 4 GPUs - If running on p2.xlarge, change to just 0.  
export DMLC_WORKER_ID=0 # your worker id
export DMLC_NUM_WORKER=1 # you only have one worker
export DMLC_ROLE=worker # your role is worker

# the following value does not matter for non-distributed jobs 
export DMLC_NUM_SERVER=1 
export DMLC_PS_ROOT_URI=10.0.0.1 
export DMLC_PS_ROOT_PORT=1234 

export EVAL_TYPE=benchmark 
python /usr/local/byteps/launcher/launch.py \
       /usr/local/byteps/example/pytorch/start_pytorch_byteps.sh \
       --model resnet50 --num-iters 1000000      
```



------------------------------------------------------------------
To run on multiple GPUs. Follow all steps above and then do:
1. Change the eval type to some classification task
```
export EVAL_TYPE=mnist
```

2. Comment out line 109 in `/usr/local/byteps/example/pytorch/train_mnist_byteps.py`
`# bps.broadcast_optimizer_state(optimizer, root_rank=0)`

3. Change line 132 in `/usr/local/byteps/example/pytorch/train_mnist_byteps.py` to 
`tensor = torch.tensor(val)`

4. Then run 
```
python /usr/local/byteps/launcher/launch.py \
       /usr/local/byteps/example/pytorch/start_pytorch_byteps.sh
```
------------------------------------------------------------------
Maybe useful for distributed training
```
docker-machine create --driver amazonec2 \
                      --amazonec2-region us-east-1 \
                      --amazonec2-zone e \
                      --amazonec2-ami ami-XXXXXX \
                      --amazonec2-instance-type p2.xlarge \
                      --amazonec2-vpc-id vpc-XXXXX \
                      --amazonec2-access-key XXXXXXXX \
                      --amazonec2-secret-key XXXXXXXXX \
                      aws01
```

sudo apt-get update && sudo apt-get -o Dpkg::Options::="--force-overwrite" install -y --no-install-recommends linux-headers-generic dkms cuda-drivers

