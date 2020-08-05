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

After understanding the encoder-decoder architecture using Show and Tell architecture, we explored the [Attention-on-Attention Net](https://arxiv.org/pdf/1908.06954.pdf) and [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) architectures.









