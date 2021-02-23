from __future__ import print_function

import subprocess
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.optim.lr_scheduler import StepLR
import numpy as np

import matplotlib.pyplot as plt


import attribution_methods
from  model import Net


def main():

    # Experiment 1: Predict at different occlusion levels 10%-90% in increments of 20% and 782 pixels for MNIST
    occ = ((np.arange(0, 1, 0.2)+0.1)*784).astype(np.int)
    #with open("test.txt", "w") as f:
    #    f.write("Accuracy\tReplacement\tOcclusion\tAttribution\tVersion\tClassifier\n")
    #with open("test.txt", "a+") as r:
    #    r.read()
    #    r.write("1\t1\t1\t1\t1\t1")
    
    #with open("test.txt", "r") as r:
    #    print(r.read())
    
    occ = np.append(782, occ)
    print(occ)
    #input(occ)
        
    for i in range(len(occ)):
        call = ["python", "src/experiments_mnist.py", "--num-classes", str(2), "--test-batch-size", str(64),
            "--occlude", str(occ[i]), "--attribution-method", "grad_orig", "--ROAR", str(0), "--epochs", str(5),
            "--classifier-type", "softmax", "--replacement-type", "dataset_min", "--model-path", "model_data/mnist_models/logsoft2/mnist1.pt", "model_data/mnist_models/logsoft2/mnist2.pt", "model_data/mnist_models/logsoft2/mnist3.pt", "model_data/mnist_models/logsoft2/mnist4.pt", "model_data/mnist_models/logsoft2/mnist5.pt"]   

        subprocess.run(call)





























if __name__ =="__main__":
    main()
