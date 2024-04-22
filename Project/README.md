# Identification of Frost in Martian HiRISE Images

## DSCI 552 Project: Machine Learning for Data Science

## Summary:
- Conducted exploration and pre-processing on a collection of 119,920 image tiles extracted from Martian HiRISE images, categorizing them into frost and background for training, validation, and testing purposes.
- Developed a binary classification model using a three-layer convolutional neural network (CNN) paired with a dense layer. Utilized various image augmentation techniques to enhance model robustness and measured performance through metrics such as precision, recall, and F1 score.
- Implemented transfer learning using established models (EfficientNetB0, ResNet50, and VGG16), which were pre-trained on the ImageNet dataset, to improve feature extraction. The performance of this approach was then compared to the original CNN + MLP model to assess the benefits of transfer learning in this context.

## Key Aspects:
This project involves developing a machine learning classifier to identify frost on Martian terrain using images from HiRISE. The classifier will be built using the Keras library in Python. The data, consisting of 214 subframes and 119,920 tiles annotated as either 'frost' or 'background', will be used in a binary classification task.

Key Components of the Project:

A. Data Exploration and Pre-processing:
The dataset includes images (png files) and labels (json files) organized by subframes, which are further sliced into 299x299 pixel tiles.
The tiles are classified into two categories, and an enhanced version of train, test, and validation splits will be provided.

B. Training CNN + MLP (Convolutional Neural Network + Multi-Layer Perceptron):
The task involves building a three-layer CNN followed by a dense layer. This includes the use of ReLU activation, softmax function, batch normalization, a 30% dropout rate, L2 regularization, and the ADAM optimizer.
Image augmentation techniques like cropping, zooming, rotating, and adjusting contrast will be applied.
The network will train for at least 20 epochs with early stopping based on validation error, and performance metrics like Precision, Recall, and F1 score will be reported.

C. Transfer Learning:
Given the relatively small size of the image dataset, transfer learning will be employed using pre-trained models such as EfficientNetB0, ResNet50, and VGG16.
Only the last fully connected layer of these pre-trained models will be trained, freezing the earlier layers. The features extracted by the pre-trained models will be used to train the replacement layers.
Similar augmentation, training strategies, and evaluation metrics as the CNN + MLP model will be applied, with a specific batch size recommendation of 8.

D. Comparative Analysis:
After training both models, their results will be compared in terms of Precision, Recall, and F1 score. An analysis will be provided to explain the differences in performance between the traditional CNN + MLP approach and the transfer learning approach.


This comprehensive project aims to leverage advanced deep learning techniques to understand the seasonal frost cycle on Mars and its implications for the planet's climate and surface evolution.







