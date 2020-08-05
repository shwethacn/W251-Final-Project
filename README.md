# Alexa! What do you See?
* A voice activated automated image captioning system on VizWiz-Captions</i>

* <b>Team:</b> Padmavati Sridhar, Shaji Kunjumohamed, Shwetha Chitta Nagaraj

## Goal:
Design and implement a voice activated image captioning system on the [VizWiz-Captions dataset](https://vizwiz.org/tasks-and-datasets/image-captioning) harnessing the power of Edge and Cloud Computing.

<b>Paper:</b> 
<b>Presentation:</b>

## Abstract:
Using a person’s Alexa device or app, he/she asks Alexa “Alexa! What do you see?” while pointing his/her cellphone to an object or scene that needs captioning.  Alexa then interacts with edge device(Jetson TX2) to get the picture from camera. Jetson then uses an image captioning model trained on the cloud to generate a caption for the picture. Jetson sends the caption text back to Alexa which is then told aloud to the user. 

## Overall Architecture:

![overall_arch](https://github.com/shwethacn/W251-Final-Project/blob/master/imgs/overall_arch.JPG)

![edge_arch](https://github.com/shwethacn/W251-Final-Project/blob/master/imgs/edge_arch.JPG)

## Dataset:
VizWiz-Captions dataset was curated by University of Texas, Austin by sourcing pictures taken from the visually impaired and captioning them using Amazon Mechanical Turk.
The dataset consists of over 40k images with captions ranging from 1- 5 captions per image. 

The Exploratory Data Analysis(EDA) of the dataset along with the file downloads can be found under: [VizWiz_EDA.ipynb](https://github.com/shwethacn/W251-Final-Project/tree/master/EDA)

## Image Captioning Models:
We explored some of the popular amongst past and present State-of-the-Art Image Captioning Architectures and techniques to shortlist the models which we will run our experiments with modeling the captioning system on the VizWiz-Captions dataset.

After understanding the encoder-decoder architecture using Show and Tell architecture, we explored the [Attention-on-Attention Net](https://arxiv.org/abs/1908.06954) and [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) architectures.

The code for the model implementations can be found here:

* [Attention-on-Attention Net](https://github.com/shwethacn/W251-Final-Project/tree/master/AoA)
* [Show, Attend and Tell](https://github.com/shwethacn/W251-Final-Project/tree/master/Show_Attend_Tell)

## Evaluation Results:

The results of the inference on <b>test</b> split is tabulated below. The evaluation is based on [CIDEr-D](https://arxiv.org/abs/1411.5726) score.

#### Examples of Generated Captions:

## Edge Inference:

We used the Show, Attend and Tell model, trained on VizWiz-Captions and finetuned with MS COCO dataset as the final model for inference on Jetson TX2. Even though the AoA Net models had higher CIDer-D scores, these models were quite dense to be used in Jetson's limited memory and GPU space to obrain an efficient inference mechanism. Moreover the complete image features needed for AoANet could not be extracted using bottom-up extraction. 

Here are the demo videos of the end product in actions:

<b>Insert the video links here </b>

## Conclusion:

The team was successfully able to design and implement an end-to-end system using Cloud for training an image captioning model based on Show , Attend and Tell and successfully use Edge device(Jetson TX2) to do inference to caption images coming from a user’s camera through a voice activated device: Amazon Alexa. The inference time is around: <b>insert time</b>

This could probably be the first such system developed to help the visually impaired using real data sourced from them using the ViZWiz-Captions dataset. 












