# Dogs vs. Cats

## Overview

This repository contains the python file containing a Convolutional Neural Network and a link to training data to as a submisison to 'Dogs vs. Cats', a Kaggle Competition. The CNN is built using [Keras](https://keras.io/), a high-level neural networks API, written in Python which can be run using a Theano or Tensorflow backend. The dataset being used for the purpose of this project is the Asirra (Animal Species Image Recognition for Restricting Access) dataset. The IDEs I would suggest to use for the purpose of this project would be [Spyder](https://github.com/spyder-ide) or [Jupyter](https://jupyter.readthedocs.io/en/latest/install.html).

## Dependencies

In order to run the CNN, you'll need to install the following libraries and APIs.
* **[TensorFlow](https://www.tensorflow.org/install/) version 1.0 or later for the backend**
* **[Theano](http://deeplearning.net/software/theano/) can also be used for the backend**
* [Spyder](https://pythonhosted.org/spyder/)
* [Keras](https://keras.io/)
* [NumPy](https://docs.scipy.org/doc/numpy/user/install.html)
* [Jupyter](https://jupyter.readthedocs.io/en/latest/install.html)

Install the missing dependencies using pip
~~~~
pip install tensorflow theano spyder keras numpy jupyter
~~~~

## Usage

After installing the dependencies, use `cd` to navigate into the desired directory on your machine and launch Jupyter by entering
    ```
    jupyter notebook
    ```
or by invoking [Spyder](https://pythonhosted.org/spyder/) on your machine.

## Acknowledgements

The classifier can score above 90% accuracy on this task. It is possible for the model to score and accuracy near 99% using data augmentation, batch normalization, pre-trained models/weights such as VGG, InceptionV3 etc. After training your model, try making a single prediction using [this](https://github.com/yashtawade/Dogs-vs-Cats/blob/master/catDogTest.png) image and check out the result.
