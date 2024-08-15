# Online Federated Learning on Distributed Unknown Data Using UAVs
This repo is the pytorch implementation of paper "Online Federated Learning on Distributed Unknown Data Using UAVs".

## Prerequisites

```bash
nvidia-smi	# NVIDIA GeForce RTX 4090
nvcc -V 	# cuda-12.1
python -V	# Python 3.10.10
pip list	# torch 2.0.0+cu118, torchvision 0.15.1+cu118, GPy 1.10.0
```

The estimation of the reward function requires the GPy module. (https://sheffieldml.github.io/GPy/)

## Preparing dataset and model

You can specify the datasets, models and other hyperparameters in `./utils/options.py`. For any new dataset, you first need to download the dataset and run `./data/partition.py` to construct a distributed dataset. To use a specific model on a new dataset, you can add settings in the `./model` folder and modify `./model/build_models.py` to add the model. Currently used models include ResNet-18, CNNs, AlexNet and so on.

## OFL-UD^2

To execute the command, just run `./OFL.py`, and you will get the change of test accuracy during training. If you need visual output, you can modify the `log` parameter and save the training process to the corresponding file and plot it.
