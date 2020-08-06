### Details of the Attention-on-Attention Net(AoA Net) code structure and instructions

The [baseline code](https://github.com/Yinan-Zhao/AoANet_VizWiz) was provided by the VizWiz Challenge organizers and is cloned [here](https://github.com/Yinan-Zhao/AoANet_VizWiz/tree/76c260940cdbcdf04e7cace09e015e85cf8870c2).

The training for the AoA Net models was done on Amazon EC2 on a p2.xlarge instance using <i>Deep Learning AMI (Ubuntu 18.04) Version 27.0</i>. Pytorch 1.4 with CUDA 10.1 was used with Python 3.6.


#### Download the VizWiz/MS COCO Data

- Download the VizWiz annotation files for training using [VizWiz data downloads.ipynb](https://github.com/shwethacn/W251-Final-Project/tree/master/AoA/scripts)
  - This downloads the pre-processed COCO vocabulary which is used for fine-tuning the VizWiz model later. 
  - Also downloads the annotation files for VizWiz-Captions(train, val and test split)
  
#### Data pre-processing

The details are given in the [baseline repo](https://github.com/Yinan-Zhao/AoANet_VizWiz/tree/76c260940cdbcdf04e7cace09e015e85cf8870c2/data).
But some of the scripts and instructions have been modified. The modified scripts can be found [here](https://github.com/shwethacn/W251-Final-Project/tree/master/AoA/scripts).

Here the annotation files are preprocessed to remove any pre-canned and rejected captions from the train and val annotation files. For the test split, dummy captions are added to make them compatible with the dataloader used in the training process.

- <b>Extract meta data for the images and build vocabulary</b>

   Run the following script which will call prepro_labels.py to map all words that occur <=5 times to a special `UNK` token and create a vocabulary for the remaining words.
   ```
   $ python3 scripts/prepro_labels_vizwiz.py
   ```
   For VizWiz-Captions, the image and vocabulary information are dumped into <i>data/vizwiztalk.json</i> and discretized caption data are dumped into <i>data/vizwiztalk_label.h5</i>. For VizWiz-Captions + MSCOCO-Captions, they are dumped into <i>data/vizwiztalk_withCOCO.json</i> and <i>data/vizwiztalk_withCOCO_label.h5</i>. The processed files for VizWiz-Captions are used for training from scratch and those for VizWiz-Captions + MSCOCO-Captions are used for fine-tuning.
   
- <b>Extract image features using Bottom-up model</b>

  The image features containing objects, attributes and bounding boxes are extracted using Bottom-up model from [Bottom-up To-down model](https://github.com/peteanderson80/bottom-up-attention) in the baseline model.
  However, this model was built on an older version of Caffe on Ubuntu 14.4 due to which the libraries could not be compiled. There are other sources where users have tried to replicate the caffe model, but were only able to extract the objects and bounding boxes without the attributes. One such model available is [Facebook AI's Detectron2](https://github.com/facebookresearch/detectron2). This has some built in support for Caffe. So used this [repo]( https://github.com/MILVLG/bottom-up-attention.pytorch) to extract the features. However, only with objects and bounding boxes features. Sigh! The repo is cloned [here](https://github.com/MILVLG/bottom-up-attention.pytorch/tree/051058d2daee42ffa878e72be6e892ef6e991ef6) and here are the instructions to extract the features.
  
  - Under the cloned submodule, do the following:
      - 1. Installation
          - a. Compile the Detectron2 libraries
          ```
          cd detectron2
          pip install -e
          ```
          - b. Install and Compile apex
          ```
          cd  apex
          python3 setup.py install 
          
          cd ..
          # install rest of modules
          python3 setup.py build develop
          ```
    - 2. Feature Extraction
    
         Download the Faster R-CNN pretrained model from [here](https://github.com/MILVLG/bottom-up-attention.pytorch/tree/051058d2daee42ffa878e72be6e892ef6e991ef6)(with the caffe k=[10,100] mode) into <i>/inference directory</i>.
         ```
         bottom-up-attention.pytorch/inference$ ls -ltr 
         -rw-rw-r-- 1 ubuntu ubuntu 390622 Jul 27 20:56 bua-caffe-frcn-r101_with_attributes.pth
         ```
         
         Start the extraction:
         ```
         python3 extract_features.py --mode caffe --config-file configs/bua-caffe/extract-bua-caffe-r101.yaml --image-dir images --gt-bbox-dir log --out-dir out_dir  --resume
         ```
         
         The objects and bounding boxes features will be stored under <i>out_dir</i> in numpy format(.npz).
         
         The notebook to visualize the output images with extracted features is under [visualize_bottom_up.ipynb](https://github.com/shwethacn/W251-Final-Project/tree/master/AoA/scripts).
         
         Since this doesn't extract the attributes of objects in images needed for AoA Net model, the pre-extracted features are downloaded from [here](http://ivc.ischool.utexas.edu/VizWiz_final/caption/AoANet_VizWiz/data/tsv.zip) and put under <i>data/tsv</i> directory.
         ```
         cd tsv
         wget  http://ivc.ischool.utexas.edu/VizWiz_final/caption/AoANet_VizWiz/data/tsv.zip
         unzip tsv.zip
         rm tsv.zip
         ls
         # Following files will be extracted. Tab separated files for train-val and test split
         VizWiz_resnet101_faster_rcnn_genome_test.tsv.1  VizWiz_resnet101_faster_rcnn_genome_trainval.tsv.2  VizWiz_resnet101_faster_rcnn_genome_trainval.tsv.3
         ```         
         Once extracted, run the following which will create 3 directories: vizwizbu_fc(objects), vizwizbu_box(bounding boxes), vizwizbu_att(attributes) with files for each image in the dataset.
         ```
         python3 scripts/make_bu_data_vizwiz.py
         ```
         
    - 3. Pre-process the dataset for n-gram on the captions which will be used to get the cache for calculating Cider score for SCST training phase:
    ```
    # For training from scratch
    python3 scripts/prepro_ngrams_vizwiz.py --dict_json data/vizwiztalk.json --output_pkl data/vizwiz-train
    
    # This will create under data/:
    -rw-rw-r-- 1 ubuntu ubuntu 74114853 Jul 23 19:29 vizwiz-train-words.p
    -rw-rw-r-- 1 ubuntu ubuntu 78920979 Jul 23 19:30 vizwiz-train-idxs.p
    
    # For fine-tuning the model
    python3 scripts/prepro_ngrams_vizwiz.py --dict_json data/vizwiztalk_withCOCO.json --output_pkl data/vizwiz-train-withCOCO
    
    # This will create under data/
    -rw-rw-r-- 1 ubuntu ubuntu 74114853 Jul 23 19:29 vizwiz-train-words.p
    -rw-rw-r-- 1 ubuntu ubuntu 78920979 Jul 23 19:30 vizwiz-train-idxs.p
    ```
    
#### Training:
The training is first done from scratch using VizWiz-Captions and then fine-tuned using MS COCO dataset.

The training scripts are under the baseline repo but some of the scripts are changed where necessary to change the parameters used for training and can be found [here](https://github.com/shwethacn/W251-Final-Project/tree/master/AoA/scrips/).

  - 1. Training from scratch(AoA-Scratch Model)
    ```
    # under scripts directory
    $ CUDA_VISIBLE_DEVICES=0 sh train_vizwiz.sh
    ```
    The pickle and Tensorboard files are generated under <i>log</i> directory.
    
  - 2. Run evaluation on test split
    ```
    $ CUDA_VISIBLE_DEVICES=0 sh eval_scratch.sh
    ```
 - 3. The evaluation results are generated under <i>vis</i> directory and is available under [VizWiz_scratch_rl.json](https://github.com/shwethacn/W251-Final-Project/tree/master/AoA/inference).
 
   This file is submitted to the competition on [Eval AI server](https://evalai.cloudcv.org/web/challenges/challenge-page/525/my-submission) to get the scoring results.
   
 - 4. Train a fine-tuned model using MS COCO(AoA-FineTuned Model):
   ```
   $ CUDA_VISIBLE_DEVICES=0 sh finetune_vizwiz.sh
   ```
   
- 5. Evaluate the fine-tuned model
  ```
  $ CUDA_VISIBLE_DEVICES=0 sh eval_finetune.sh
  ```
  
- 6. Results are available under [VizWiz_finetune_rl.json](https://github.com/shwethacn/W251-Final-Project/tree/master/AoA/inference).


    


         

         
         

  

   

  
