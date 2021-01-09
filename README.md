# [On the Texture Bias for Few-Shot CNN Segmentation]

This repository contains the code for our WACV'21 paper 'On the Texture Bias for few-shot CNN Segmentation'.

## Prerequisties and Run
This code has been implemented in python language using Keras libarary with tensorflow backend and tested in ubuntu OS, though should be compatible with related environment. following Environement and Library needed to run the code:

- Python 3
- Keras version 2.2.0
- tensorflow backend version 1.13.1


## Run Demo
The implementation code is availabel in Source Code folder.</br>
1- Download the FSS1000 dataset from [this](https://drive.google.com/open?id=16TgqOeI_0P41Eh3jWQlxlRXG9KIqtMgI) link and extract the dataset.</br>
2- Run `Train_DOGLSTM.py` for training Scale Space Encoder model using k-shot episodic training. The model will be train for 50 epochs and for each epoch it will itterate 1000 episodes to train the model. The model will saves validation performance history and the best weights for the valiation set. It also will report the MIOU performance on the test set. The model by default will use VGG backbone with combining Block 3,4 and 5 but other combination can be call in creatign the model. It is also possible to use any backbone like Resnet, Inception and etc.... </br>
3- Run `Train_weak.py` for training Scale Space Encoder model using k-shot episodic training and evaluatign on the weak annotation test set. This code will use weaklly annotated bouning box as a label for the support set on test time.

Notice: `parser_utils.py` can be used for hyper parameter setting and defining data set address, k-shot and n-way.

