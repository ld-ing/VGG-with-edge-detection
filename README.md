# VGG-with-Edge-Detection

Jan. 2017

Final Project for CSC 577 (Advanced Topics in Computer Vision)

**Improving Fine-grained Image Classification with Edge Detection**

By Li Ding

## Introduction

This work is based on Python 2 and Keras library with TensorFlow backend. Data used in this work is a public image classification dataset -- Food-101, provided by Bossard et. al. in ECCV 14.

edge_trans.py is to transform original images into edge images.

VGG16_ImgNet.py is the baseline VGG-16 model of which the weights are pretrained on ImageNet.

VGG16_edge.py is the VGG-16 model with edge images as input. Weights are also pretrained on ImageNet.

Merge-at-first.py and Merge-at-last.py are the two proposed structures in the paper. Both of them are based on the models in VGG16_ImgNet.py and VGG16_edge.py, trying different ways to merge them.

Report.pdf is the final report for the project.

All above credit to Li Ding, all rights reserved.
