import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--createEnv', action="store_true", default=False)
parser.add_argument('-b', '--buildModel', action="store_true", default=False)
parser.add_argument('-cr', '--colabRequirements', action="store_true", default=False)
parser.add_argument('-ce', '--colabCondaEnv', action="store_true", default=False)
args = parser.parse_args()


# Create conda environment
if args.createEnv:
    os.system('conda env create -f environment.yml')

# Build package for model directory
if args.buildModel:
    os.system('python setup.py bdist_wheel --universal')

# Install requirements on colab
if args.colabRequirements:
    os.system('!pip install -r requirements_colab.txt')

# Create conda environment on colab
if args.colabCondaEnv:
    os.system('!pip install -q condacolab')
    import condacolab
    condacolab.install()
    os.system('!conda env create -f environment.yml')
