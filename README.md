# ANOMALOUS  BEHAVIOUR  IN  LOSS-GRADIENT  BASED INTERPRETABILITY METHODS

Code for work submitted in the RobustML workshop at ICLR 2021 (https://sites.google.com/connect.hku.hk/robustml-2021/home?authuser=0)

## Setup

Create a python 3 venv and activate the environment (https://docs.python.org/3/library/venv.html)
```
$ python3 -m venv venv
$ source venv/bin/activate
```

Install the required packages with

```
$ pip install -r requirements.txt
```
  

## MNIST experiments

### Train models

The first step is to train MNIST models. The main change from the pytorch tutorial for MNIST is the data loader loads the entire dataset as a numpy array making it easier to manipulate the input for the attribution methods.

The typical MNIST model with 10 classes and logsoftmax output as in the pytorch tutorial is trained with the following code:

```
$ python src/train_mnist.py --num-classes 10 --classifier-type softmax --model-path YOUR_MODEL_PATH --save-model
```

For the two class case of classifiying between 0 and 1 with the logsoftmax output the code is:
```
$ python src/train_mnist.py --num-classes 2 --classifier-type softmax --model-path YOUR_MODEL_PATH --save-model
```
For the two class case of classifying between 0 and 1 with the sigmoid output the code is:

```
$ python src/train_mnist.py --num-classes 2 --classifier-type sigmoid --model-path YOUR_MODEL_PATH --save-model
```

Note that the sigmoid output model only works for the binary classification task.


Other parameters:

``` --batch-size``` Batch size for training, default is 64
``` --test-batch-size``` Batch size for evaluating on test data, default is 1000

### Evaluate on occluded data

Running the ```src/experiments_mnist.py``` file computes the test accuracy depending on the attribution method,
replacement value and ranking order.

```
$ python src/experiments_mnist.py --num-classes 2 --classifier-type sigmoid --test-batch-size 64 --occlude 84 
                                  --attribution-method grad_orig --ROAR 1 --replacement-type mean 
                                  --model-path YOUR_MODEL_PATH --save-images YOUR_IMAGE_PATH
```

The attribution used to occlude the input is determined by the ```--attribution-method``` command. The different
methods used in the paper are:
* abs_grad
* grad_orig
* grad_inp
* random

The replacement value is determined by the ```--replacement-type``` command. The different replacement types are:
* mean
* image_min
* image_max

The ranking order of either highest to lowest or lowest to highest is determined by ```--ROAR```. If the value is 
1 then the gradients are ranked from most positive to most negative and the inputs are removed from the most positive
side. If the value is 0 then the gradients are ranked from most negative to most positive and the inputs are removed 
corresponding to the most negative values. 

The ```--occlude``` command determines the number of pixels to occlude. For MNIST this is in the range of 1 to 783.
As with the training process ```--num-classes``` and ```--classifier-type``` is used to select the type of model
to evaluate. 


## Singing voice detection

### Data preparation
The jamendo dataset can be downloaded from (https://zenodo.org/record/2585988). The folder structure we used for the dataset:

* repo
  * datasets
    * jamendo
      * audio
      * labels
      * filelists
        * train
        * test
        * valid
  * src

### Train models

```
$ python src/train_singing.py YOUR_MODEL_PATH --cache-spectra YOUR_SPECTRA_PATH --validate 
```

```--cache-spectra``` indicates the path to store or retrieve the computed log mel spectrograms for. 

Compute predictions on normal data using the command:
```
$ python src/predict.py YOUR_MODEL_PATH YOUR_PREDICTION_PATH --cache-spectra YOUR_SPECTRA_PATH
```
The predictions are saved as a .npz file and then you can evaluate your model using the command:
```
$ python src/eval.py YOUR_PREDICTION_PATH --auroc
```

### Evaluate on occluded data

To evaluate on occluded data use the command:
```
$ python src/ROAR_predict.py YOUR_MODEL_PATH YOUR_PREDICTION_PATH --cache-spectra YOUR_SPECTRA_PATH
                            --ROAR 1 --occlude 920 --attribution-method grad_orig --replacement-type mean
                            --loss-grad-save YOUR_GRAD_PATH --lossgradient YOUR_GRAD_MODEL_PATH
```
The model to compute the loss gradient is given by the command ```--lossgradient```. For this workshops experiments
this model is the same as the model we are computing the predictions for. However, you can change it to another model
with the same architecture. 

Since loss gradient computation is time intensive the ```--loss-grad-save``` command lets you select a folder to load or
save the loss gradients for so that running evaluations after the first time is much quicker.

The commands that are identical to MNIST have the same options available so refer to the MNIST experiments above.
