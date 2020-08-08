# Show Attend and Tell - Training in the Cloud

The starter code was taken from this Github repo [baseline code](https://github.com/fg91/Neural-Image-Caption-Generation-Tutorial).  


## List of files

* `Dockerfile.nvidia`  
This file is used to bring up the docker container in the cloud, for training.

* `Prepare_data.ipynb`  
Notebook used for pre-processing data in the Cloud. 

* `Training_coco_vizwiz_combined.ipynb`   
Notebook used for training model in the cloud 


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

| ![Train-Validation Curve](https://raw.githubusercontent.com/shwethacn/W251-Final-Project/master/Show_Attend_Tell/imgs/251_cloud_arch.jpg) | 
|:--:| 
| The The network architecture at the cloud is captured in the above figure. The figure shows a GPU machine with 2T  local SSD and another NFS machine with a 2T SSD, which can be mounted across all machines as an NFS share. All data is stored in NFS machine, which is persistent. The GPU machine can be torn down and built up, based on demand. This methodology is followed to keep costs down. |

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

refer to the notebook `Prepare_data.ipynb` for more details.

| ![Train-Validation Curve](https://raw.githubusercontent.com/shwethacn/W251-Final-Project/master/Show_Attend_Tell/imgs/251_steps.jpg) | 
|:--:| 
| The various steps involved in training in cloud and creating a model for inference at the edge is captured in the above figure |

## Data wrangling, cleaning up and filtering dataset

This step is accomplished by using Dask framework. The Steps are outlined in the notebook Prepare.ipynb available in this folder. All processed data is saved as python pickle files and handed over to downstream processes.

## Dataset argumentation

As a first step, training and validation using  vizwiz data set alone were attempted.  However, this yielded poor results in terms of BLEU scores and model had problems in converging. For getting better results, VizWiz dataset training set is mixed with COCO training data set and Validation set is mixed with COCO validation set and used for developing the model.

## Model and Training heighlights

The final model we used for inference is based on ‘Show attend and Tell’ paper.  In this model, we start with a vector representing the input image. The vector is generated by an encoder, which is just a  Resnet34 pre-trained CNN model. We take this vector through a RNN, which gives out a vector of activations. We train this RNN in such a way that this activation represents a sentence that captures meaning and structure of the image. Since we have deeper networks, we used GRU  in place of RNN to make training easy, along with standard regularization techniques. This is the decoder that we use. We used the BLEU metric to Measure how good the translation is, with respect  to the target caption.


For training, We used a technique called teacher forcing.  We also used attention and  Beam search to obtain good captions out of images. 
The following Parameters affected final model architecture 
* The training time at the cloud 
* Jetson RAM utilization
* Jetson GPU activity at inference time  

We selected simpler models to make it fit in Jetson’s RAM. Fast ai framework was used for model development - this framework  provided high level components to build the training and inference machinery.

Since each image in the dataset had multiple captions, we randomly selected a caption per image per epoch.  This resulted in low BLEU scores while training, but yielded very less overfitting.  We trained using batches of epochs and after each batch we ran ‘One cycle policy’ and adjusted initial learning rate for that batch, which resulted in faster convergence, Again, many techniques were used to improve training time such as - pickling of image files, adjusting number of workers. We trained over an nfs mount and only frequently accessed data was transferred to GPU machine. This helped in efficient data and resource management.


## Training time

The initial training time was 30 minutes. We employed the following strategies,  to improve training time
Converted image files into pickle files for faster loading
Moved the frequently accessed data to local disk (The pickle files)
Used more workers in data loader.  The parallel workers fetch the data from the disk in advance in a pipelined fashion giving better throughput
These techniques helped us in bringing training time to 6 minutes per epoch. We were able to do more epochs and get a better score.

## Usage of one cycle policy

We extensivily used one cycle policy for to find out optimum learning rate. In one cycle policy, it will do a mock training by going over a large range of learning rates, then plot them against the losses. We will pick a value a bit before the minimum, where the loss still improves. We did our training using batches of epochs. After each batch of epochs, the learning rate changes. We ran one cycle policy to find out the optimum learing rate for the next batch. 

| ![Train-Validation Curve](https://raw.githubusercontent.com/shwethacn/W251-Final-Project/master/Show_Attend_Tell/imgs/251_train_val_curve.jpg) | 
|:--:| 
| The training validation curve for the first batch of epochs are shown above. |

| ![Initial Learing rate setting](https://raw.githubusercontent.com/shwethacn/W251-Final-Project/master/Show_Attend_Tell/imgs/251_initial_lr.jpg) | 
|:--:| 
| The figure shows the setting for the initial learning rate is (shown above). The initial learningrate is set as 1e-03 as the point of steepest descent is 1e-03 |

| ![Second learning rate](https://raw.githubusercontent.com/shwethacn/W251-Final-Project/master/Show_Attend_Tell/imgs/251_second_lr.jpg) | 
|:--:| 
| The figure above shows setting up of second learning rate. It is setup as 1e-04 |

| ![No overfitting](https://raw.githubusercontent.com/shwethacn/W251-Final-Project/master/Show_Attend_Tell/imgs/251_no_overfitting.jpg) | 
|:--:| 
| The figure above shows that there is no overfitting |

## Setting up model for inference

The inference was attempted on GPU and CPU. It has been observed that predictions run on CPU were of a low quality than the GPUPU. The model structure and parameters were saved. The model is saved as a reusable library. The parameters were saved using Pytorch native format. In the EDGE device, the model was re-constructed and trained parameters were loaded back on the model to do inferencing.   
* For testing the model, it was deployed as a flask service that accepts jpeg images. Images were sent to the model and the model returned corresponding captions.  
* At the rate of 5 fps, continuous frame per frame inferencing was attempted on a streaming video. The captions were superimposed on top of each frame of video, just like YOLO demo. 
* We reduced the CNN from Resnet101 to Resnet34 to accommodate the model in Jetson RAM. No degradation in model performance was observed, even though the model became simpler. 
