# Show Attend and Tell - Training in the Cloud

The starter code was taken from this Github repo [baseline code](https://github.com/fg91/Neural-Image-Caption-Generation-Tutorial).  


## Resources

The training was done on IBM cloud with following resources

* An Ubundu 18.04 LTS machine with v100 GPU, 8 CPU and 128G RAM, 2T SSD for training.  
* An Ubundu 18.04 LTS machine with 2 CPU and 32G RAM, 2T SSD as NFS server

## Tools

Following tools were used for traing

* Dask framework  
* Pytorch Docker container from nVidia
* Python Multiprocessing utility.
* [fast.ai](https://github.com/fg91/Neural-Image-Caption-Generation-Tutorial) framework

## General work flow

Since data need to be transferred in and out of machines over the network frequently, an NFS file share server was set up in the cloud. All the files and data was stored in this machine.  Initially "rsync" was used to transfer data. Since "rsync" turned out to be a bit slow, we moved a to a webserver based mechanism. Used "wget" to move data and offered better speed.

### Step1 : set up NFS server and client

The steps for setting up NFS server client is outlined [here](https://vitux.com/install-nfs-server-and-client-on-ubuntu/). The Client machine here is the v100 GPU machine. This workflow was used to minimize the expenses associated using the machine with GPU. Once the training is done, the GPU machine is cancelled, but data is safely saved in the shared drive. 

### Step2 : Bring up the GPU v100 machine 

The GPU v100 machine is used for training. A Docker container need to be setup before training. Below are the steps for bringing up Docker container. 

1. Order a v100 machine. Below is the command used for same :

```bash
ibmcloud sl vs create --datacenter=dal10 --hostname=v100 --domain=shaji.com --os=UBUNTU_18_64 --flavor AC2_8X60X100 --billing=hourly --san --disk=100   --network 1000
```

2. Install Graphics drivers

```bash

% sudo apt update
% sudo add-apt-repository ppa:graphics-drivers
% sudo apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
% sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list'
% sudo bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda_learn.list'
% sudo apt update
% sudo apt install cuda-10-1
% sudo apt install libcudnn7
```

3. Install docker

```bash

% apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    software-properties-common
	
% curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

% sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"	

% apt-get update


% docker run hello-world

```

4. Install nvidia-Docker

```bash

% curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
% curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
% apt-get update
```
5. Install the 2T disk, Format and mount disk. It is important to move Docker directory to the mounted SSD

```bash
% fdisk -l
% mkdir -m 777 /data1
% mkfs.ext4 /dev/xvdc
% cat /etc/fstab

# edit /etc/fstab and all this line:
/dev/xvdc /data                   ext4    defaults,noatime        0 0

% mount /data1

% service docker stop
% cd /var/lib
% cp -r docker /data1
% rm -fr docker
% ln -s /data1/docker ./docker
% service docker start
```

6. Bring up the pytorch container with associated software. The Docker file is provided with in this folder. Folowing are the commands for the same

```bash
% docker build -f Dockerfile.nvidia -t model .
# This needs to be done from the /data partition
% docker run -it --rm -v "$PWD":/home -h "training" --runtime=nvidia  -p 8800:8888  --shm-size=30G model:latest bash
```

7. Bring up jupyter lab in the docker.
```bash
% jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```


### Setting up data for training

Following data need to be downloaded
1. [COCO 2014](https://cocodataset.org/#download) Dataset. Download the train/validation/test and annotations
2. [VIZWIZ dataset](https://vizwiz.org/tasks-and-datasets/image-captioning/). Download train/validation/test.

The data can be downloaded to the shared `/data` partition and `/data1` partition (local partition in the GPU machine). This is because the machine with GPU can be really slow to read data over the network. Could be up to 4X slower. The image must be converted into pickle files for improving training time. This can be accomplished using below code snippett.

```python
#!/usr/bin/env python3

import time
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count
from PIL import Image
import glob
from pathlib import Path
import pickle

parse_array = ['train2014/*.jpg', 'val2014/*.jpg']
num_processes = cpu_count() * 50

def process_image(name):
    img = Image.open(name).convert('RGB')
    pickle.dump(img, open(name.with_suffix('.jpkl'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def main():

    start = timer()

    print(f'starting {num_processes} process computations on {cpu_count()} CPU cores')
    values = []
    for objs in parse_array :
       for name in glob.glob(objs) :
           values.append(Path(name))

    with Pool(processes=num_processes) as pool: pool.map(process_image, values)

    end = timer()
    print(f'elapsed time: {end - start}')

if __name__ == '__main__':
    main()

```

## Data wrangling, cleaning up and filtering dataset

This step is accomplished by using Dask framework. The Steps are outlined in the notebook Prepare.ipynb available in this folder. All processed data is saved as python pickle files and handed over to downstream processes.

## Dataset argumentation

As a first step, training and validation using  vizwiz data set alone were attempted.  However, this yielded poor results in terms of BLEU scores and model had problems in converging. For getting better results, VizWiz dataset training set is mixed with COCO training data set and Validation set is mixed with COCO validation set and used for developing the model.


```python

```
