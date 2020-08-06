### Details of the Attention-on-Attention Net(AoA Net) code structure and instructions

The [baseline code](https://github.com/Yinan-Zhao/AoANet_VizWiz) was provided by the VizWiz Challenge organizers and is cloned [here](https://github.com/Yinan-Zhao/AoANet_VizWiz/tree/76c260940cdbcdf04e7cace09e015e85cf8870c2).

The training for the AoA Net models was done on Amazon EC2 on a p2.xlarge instance using <i>Deep Learning AMI (Ubuntu 18.04) Version 27.0</i>. Pytorch 1.4 with CUDA 10.1 was used with Python 3.6.


#### Download the VizWiz/MS COCO Data

- Download the ViZWiz annotation files for training using [VizWiz data downloads.ipynb](https://github.com/shwethacn/W251-Final-Project/tree/master/AoA/scripts)
  - This downloads the pre-processed COCO vocabulary which is used for fine-tuning the VizWiz model later. 
  - Also downloads the annotation files for VizWiz-Captions(train, val and test split)
  
  
