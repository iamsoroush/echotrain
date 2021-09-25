# EchoTrain

Training engine of EchoApp.

This repository mainly used four files for training process:
* **Config**: Specify all configs that need to be set for each following three classes to train model
* **dataset**: Perform some pre or post processes on the given dataset and create a data generator for the given dataset
* **model**: Define the structure of the model that wants to be trained and metrics that wants to be used for evaluating the model during the training process
* **training**: Train the given model based on the inputs, log the required information with mlflow and tensorboard during training, determine the best weights for the model, and store the config file

Also Inference class could be used to perform the required processing to prepare data for the model and then do the lv semantic segmentation based on the trained model

