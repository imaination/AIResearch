What modules to download on the genome center computer system.

1. TensorFlow API (includes python)
2. tensorflow model
3. pip install pillow, lxml, jupyter, matplotlib

STEP 1. To install TensorFlow: 
$ conda create -n tensorflow pip python=2.7 #or python=3.3, etc. \
$ source activate tensorflow \
$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.5.0-py2-none-any.whl \
$ source deactivate #to deactivate virtual env 

STEP 2. To install tensorflow model:
$ git clone https://github.com/tensorflow/models.git \
Check to see if you have protoc. \ 
$protoc --version 

Run below commands in the directory, models/research: 
$protoc object_detection/protos/*.proto --python_out=. \
$ export PYTHONPATH=$PYTHONPATH:pwd:pwd/slim #(with back slashes around pwd)

From models/research, run: 
$ sudo python setup.py install #this formally installs the object_detection library

STEP 3. To install pillow, lxml, jupyter, matplotlib:
$ pip install pillow 
$ pip install lxml 
$ pip install jupyter 
$ pip install matplotlib


