import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--create_env', action="store_true", default=False)
parser.add_argument('-b', '--build_model', action="store_true", default=False)
parser.add_argument('-cr', '--colab_requirements', action="store_true", default=False)
parser.add_argument('-ce', '--colab_conda_env', action="store_true", default=False)
args = parser.parse_args()


# Create conda environment
if args.create_env:
    os.system('conda env create -f environment.yml')

# Build package for model directory
if args.build_model:
    os.system('python setup.py bdist_wheel --universal')

# Install requirements on colab
if args.colab_requirements:
    os.system('!pip install -r requirements_colab.txt')

# Create conda environment on colab
if args.colab_conda_env:
    os.system('!pip install -q condacolab')
    import condacolab
    condacolab.install()
    os.system('!conda env create -f environment.yml')
