# cpVision - Senior Design Project
Developed A.I. software that utilizes advanced computer vision algorithms to improve diagnostics and facilitate more efficient planning in the field of gastroenterology

## Demo
Link to demo of models in action: https://drive.google.com/file/d/1zhqsn3N3c6vFw-fqsHPBn_AxR-88Kxxo/view?usp=sharing

(WARNING: Demo video may be graphic for some)

Note: frames utilized in the demo are from public data from the Nerthus dataset.

## Overview
With all the great developments in A.I. technologies, it is imperative that we find ways to apply them to real-world scenarios. Currently, there is an industry-wide need for advanced tools that can aid colonoscopists with detecting polyps. The cpVision project aims to conquer this task by developing a clinical support device that can assist colonoscopists with the bowel-preparation assessments portion of their procedure. cpVision seeks to remove the inter-observer variability in bowel cleanliness scoring by utilizing state-of-the-art deep learning methods to accurately score a patient’s gastro-intestinal tract. The final deliverable of the project will be a clinically-deployable device that interfaces with a colonoscopy video feed and generates real-time predictions of bowel preparation.

## Background
A colonoscopy is a medical examination where a gastroenterologist inserts a colonoscope through the rectum in order to observe the colon. This procedure is usually done to investigate the cause of intestinal symptoms and to screen for colon cancer [1]. Before the colonoscopy, patients are required to cleanse their bowels by altering their diet for a couple of days and by using a laxative to eliminate any stool [2]. A colon that is free of stool and residue will allow the physician to clearly detect any signs of colorectal cancer. Prior to the procedure, the physician will check that the colon is free of residue. If the colon is not adequately prepared, they will postpone the procedure.

In the past, colon cleanliness was assessed without any standardized method and scored using subjective terms: “excellent,” “good,” “fair,” and “poor.” [3] This led to inter-observer variability: different doctors attributed a different term to the same colon. In order to reduce this variability, the Boston University Medical Center developed the Boston Bowel Preparedness Scale (BBPS). The BBPS score works as follows:

<img width="400" alt="bbps" src="https://github.com/kevry/cpVision/assets/45439265/502ec1e4-b6ec-4e1a-8c27-f635836d627d">


## Algorithms
The main feature of cpVision is its objective bowel preparation scoring system. Our team achieves this by using Image Analysis and state-of-the-art machine learning algorithms. More specifically, we use Convolutional Neural Networks for our models. The algorithm is consistent with its scoring and has no bias.

To build the models needed for this project, we need a large dataset of labeled gastrointestinal-tracts.  Our dataset consisted of non-public images and publicly available data called The Nerthus Dataset. Furthermore, before we implemented a deep learning model, all images needed to go through a data preparation pre-process. This process involves cleaning and transforming raw data to avoid any unnecessary bias and information that can occur during training.
Image Classification w/ Deep Learning

One of our main goals for this project was to develop an algorithm that can correctly classify a patients gastro-intestional tract based on bowel cleanliness. To achieve this, we decided to implement a state-of-the-art deep learning algorithm called Convolutional Neural Networks (CNN). For our case, we will be using a CNN model to implement image classification. As stated before, our classes are as follows; 0, 1, 2, and 3 where 0 is an unprepared bowel with visible stool, and 3 is a clean bowel. For our approach, the team implemented Transfer Learning, which is a technique used in deep learning, to use pre-trained models and fine-tune it for our specific problem. In our case, we decided to use the pre-built architecture ResNet50 with pre-trained weights from the IMAGENET dataset. To fine-tune the pre-trained model, a fully connected layer was added at the end of the current architecture. For the implementation of the model, we utilized 10,569 images for training, and 1,150 for testing.  

Our model achieves an overall accuracy of 93.74% on the provided testing set. The predictions visible by the device are calculated by our model. Below is a breakdown of the accuracy for each individual class. 

<img width="500" alt="bbps_model_rate" src="https://github.com/kevry/cpVision/assets/45439265/0b285cfd-9597-42c9-a4f1-ba2b9510fdbe">

In addition to image classification, the team decided to implement an object detection model for the convenience of endoscopists. Specifically, we developed a model that detects stool in real-time. For the development process, we utilized YOLOv3, a state-of-the-art real-time object detection system. During each process, the object detection model will box any area within a frame that is classified as stool.

## Optimizing
Even with the prediction FPS and the output video FPS being disjoint, it is still desirable to maximize the prediction FPS. The bottleneck with the prediction FPS is the deployment of the ResNet models built in the Tensorflow. The ResNet models take up a significant amount of RAM when loaded. In fact, there is usually less than 500 MB left when the GUI is fully running. Since our team utilized such a computationally expensive model, we needed to adapt deployment methods. Our team utilized TensorRT, which is a high-performance framework built specifically for inference on NVIDIA products. The TensorRT plugin allows us to optimize GPU and CPU usage during runtime, unlike Tensorflow. The TensorRT increased prediction from 5 FPS to around 12 FPS.

## Acknowledgement
Members of the cpVision - Senior design team
- Kevin Delgado
- Karim Khalil
- Grayson Wiggins
- Ryan Schneider
- Zhengjian Chen
