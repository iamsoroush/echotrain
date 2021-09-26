# EchoTrain

Training engine of EchoApp.

This repository mainly used four files for training process:
* **Config**: Specify all configs that need to be set for each following three classes to train model
* **dataset**: Perform some pre or post processes on the given dataset and create a data generator for the given dataset, the following image shows these processes for a dataset
![alt text](http://gitlab.aimedic.co/soroush.moazed/echotrain/-/raw/aboutme/attachments/data_ingestion.drawio.png)
* **model**: Define the structure of the model that wants to be trained and metrics that wants to be used for evaluating the model during the training process
* **training**: Train the given model based on the inputs, log the required information with mlflow and tensorboard during training, determine the best weights for the model, and store the config file

Also, Inference class could be used to perform the required processing to prepare data for the model and then do the lv semantic segmentation based on the trained model

Four classes mainly used for training process:
* **DatasetHandler**: This class generates data generators from a given dataset 
* **PreProcessor**: Perform required processes before using the data generators in training
* **Model**: Define and compile the model that wants to be trained
* **Trainer**: Train the given model based on the inputs and store tensorboard and mlflow logs

![alt text](http://gitlab.aimedic.co/soroush.moazed/echotrain/-/raw/aboutme/attachments/training.drawio.png)

# Installation

## Conda
This command could be used to create a conda environment of the project:

`conda env create -f environment.yml`

## Colab

This command could be used to install the requirements of this project on google colab:

`!pip install -r requirements_colab.txt`