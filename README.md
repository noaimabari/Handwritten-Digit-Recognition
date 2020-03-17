# Handwritten-Digit-Recognition

This project is used to identify handwritten digits. 
I have used a dataset of images of handwritten digits, available on kaggle, to train my model.
In the model, I have used 2 dimensional CNN layers and max pooling layers to pre process the images in order to find the features to work upon. The data has been then flattened and passed into the dense neural network layers for classification. The activation function used in different layers is different. Adam optimizer is used for compiling the model with binary_cross_entropy as the loss function.
This model can achieve an accuracy of more than 98.5% on the validation data in just 5 epochs.
