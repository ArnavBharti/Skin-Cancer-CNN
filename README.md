# Skin Cancer Classification with MobileNet

This project classifies 7 different types of skin cancer using two MobileNet-based deep learning models. The models are trained and evaluated on a dataset of skin lesion images.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)

## Introduction

The goal of this project is to build and train two multi-layer convolutional neural network (CNN) architectures based on the MobileNet architecture for the classification of skin cancer. The project includes scripts for data loading and augmentation, model training, and evaluation.

## Dataset

The dataset contains 7 classes of skin lesions:
1.  **akiec**: Actinic keratoses and intraepithelial carcinoma / Bowen's disease
2.  **bcc**: Basal cell carcinoma
3.  **bkl**: Benign keratosis-like lesions
4.  **df**: Dermatofibroma
5.  **mel**: Melanoma
6.  **nv**: Melanocytic nevi
7.  **vasc**: Vascular lesions

The training set consists of 3,000 grayscale images, and the validation set contains 1,000 images, all with dimensions of 300x300.

## Usage

You can train and evaluate the models using the provided Google Colab notebook. The notebook will guide you through the following steps:
1.	Setup: Mount your Google Drive and set up the necessary libraries.
2.	Configuration: Define the hyperparameters for training.
3.	Data Loading: Load the training and validation data.
4.	Training: Train the two MobileNet-based models.
5.	Evaluation: Evaluate the models on the validation set and display the results, including training/validation plots, a confusion matrix, and per-class accuracy.

## Models

Two MobileNet-based models are implemented:
1. MobileNetModel1: A fine-tuned MobileNetV2 with a modified classifier.
2. MobileNetModel2: A fine-tuned MobileNetV3-Small with a custom classifier that includes a Mish activation function.

## Results

The performance of each model is evaluated based on the following metrics:
- Training and validation accuracy and loss curves
- Confusion matrix for the validation set
- Overall and per-class accuracy
The results are displayed in the Google Colab notebook after training and evaluation.
