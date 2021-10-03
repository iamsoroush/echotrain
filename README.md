# EchoTrain

Training engine of EchoApp.

Four classes mainly used for training process:
* **DatasetHandler**: This class is implemented in the dataset directory and creates data generators from a given dataset by using the `create_data_generators` method.
* **PreProcessor**: This class is implemented in the model directory and performs required processes before using the data generators in training and finally returns processed data in a new data generator. The following image shows these processes for a dataset:

![alt text](http://gitlab.aimedic.co/soroush.moazed/echotrain/-/raw/aboutme/attachments/data_ingestion.drawio.png)

* **Model**: This class is implemented in the model directory and define the structure of the model that wants to be trained and metrics that wants to be used for evaluating the model during the training process.
* **Trainer**: This class is implemented in the training directory and Train the given model based on the inputs and config file moreover store tensorboard and mlflow logs by using `train` method and determine the best weights for the model, and store the config file by using `export` method. 

![alt text](http://gitlab.aimedic.co/soroush.moazed/echotrain/-/raw/aboutme/attachments/training.drawio.png)

# Training

The model could be trained by using the following command:

`python train.py --experiment_dir /path/to/experiment`
 
by using the following command the program will determine to load the corresponding dataset, model, and preprocess's by using the config file in the experiment directory and then train the model based on the loaded files.

# Installation

## Conda
This command could be used to create a conda environment of the project:

`conda env create -f environment.yml`

## Colab

This command could be used to install the requirements of this project on google colab:

`!pip install -r requirements_colab.txt`

# Automation 

After cloning this project from git, the following commands could be used to create the conda environment based on the environment.yml file:

`echotrain.py -c` or `echotrain.py --createEnv`

Any time that you need to build the package for model directory the following commands could be used:

`echotrain.py -b` or `echotrain.py --buildModel`

Also, you could install the requirements needed in the colab by using the following commands:

`echotrain.py -cr` or `echotrain.py --colabRequirements`

Finally, the conda environment could be created in the colab by using the following commands:

`echotrain.py -ce` or `echotrain.py --colabCondaEnv`
