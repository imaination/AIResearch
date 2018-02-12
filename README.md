# AIResearch
Object detection using TensorFlow API


STEP 1. Installation (after installation of python 2.7)

$ pip install pillow \
$ pip install lxml \
$ pip install jupyter \
$ pip install matplotlib 

Officail Installation Site: https://www.tensorflow.org/install/install_mac#the_url_of_the_tensorflow_python_package \
Installing Tensorflow with Anaconda:\
$ conda create -n tensorflow pip python=2.7 #or python=3.3, etc. \
$source activate tensorflow \
$ pip install --ignore-installed --upgrade  https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.5.0-py2-none-any.whl 

$source deactivate #to deactivate virtual env

STEP 2. 
Install the TensorFlow model through git: \
$ git clone https://github.com/tensorflow/models.git

Check to see if you have protoc. \
$protoc --version:

Run below commands in the directory, models/research: \
$protoc object_detection/protos/*.proto --python_out=. \
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim #(with back slashes around pwd)

From models/research, run: \
$ sudo python setup.py install #this formally installs the object_detection library 
     
Step 3. Check if installation is complete. 

Go to models/research/object_detection and open jupyter notebook. \
Run all object_detection_tutorial.ipynb, you should get a pic of dogs, people, and kites with boxes. 
    
Step 4. Set up image files, xxx_label_map.pbtxt, and create_xxx_tf_record.py 

Instruction found in models/research/object_detection/g3doc/using_your_own_dataset.md

In the image directory: 
    1. test directory should have a copy of approx 10% of images with annotation data \
    2. training directory should have a copy of the rest of data with annotations   

Make a label map in the directory, models/research/object_detection/data/ \
    1. label_map should include: 

        item {
          id: 1
          name: 'xxx'
        }
    

Make a create_monkey_tf_record.py in the directory, models/research/object_detection/dataset_tools/ \
Follow the sample code found in models/research/object_detection/g3doc/using_your_own_dataset.md 

After creating create_monkey_tf_record.py file, and running it, there should be a train.record and test.record under \research/object_detection/data/ file. \
Run the file (create_monkey_tf_record.py) by, \
$ python create_monkey_tf_record.py 

Step 5. Using a pre-trained model (transfer learning)

The benefit of transfer learning is that the process is faster with less data required to train.
Need checkpoint and configuration files. 

$ wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/samples/configs/ssd_mobilenet_v1_pets.config \
$ wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz 

Alternatively, checkpoint files can be found here,  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md \
And more configuration files can be found here,  https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs

In the configuration file, fix all the path_to_be_cofigured \
num_classes to 1 \ 
num_examples to 15 \
fine_tune_checkpoint: "ssd_mobilenet_v1_coco_11_06_2017/model.ckpt \
label_map_path: "data/object-detect.pbtxt

train_input_reader: \
input_path: data/xxx_train.record \
label_map_path: data/xxx_label_map.pbtxt

eval_input_reader: \
input_path: data/xxx_test.record \
label_map_path: data/xxx_label_map.pbtxt

###just for my own reference### \
added files are: \
    1. test/train.records (data/) \
    2. create_xxx_tf.record (dataset_tools/) \
    3. xxx_label_map.pbtxt (data/) \
    4. ssd_mobilenet_v1_coco_11_06_2017 [directory] (object_detection) \
    5. ssd_mobilenet_v1_pets.config [file] (training/)

From models/object_detection, run (train_dir is where all the output will go) \
$ python train.py --logtostderr --train_dir=training/ --pipline_config_path=training/ssd_mobilenet_v1_pets.config

To check learning rate using TensorBoard: \
From models/object_detection: \
$ tensorboard --logdir='training'

STEP 6.
Testing Our Model:
Once the Lossgraph is well under 2, quit learning.

Export Graph:
From models/object_detection \
$python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path training/ssd_mobilenet_v1_pets.config \
    --trained_checkpoint_prefix training/model.ckpt-10856 \
    --output_directory xxx_inference_graph
    
Checkpoint files should be in the training directory. \
For --trained_checkpoint_prefix training/model.ckpt-10856 \ choose the model w/ largest step (the largest number after the dash)

In the object_detection directory, there should be a new directory, xxx_inference_graph. \
xxx_inference_graph should include:
     1. new checkpoint data
     2. a saved_model directory
     3. forzen_inference_graph.pb 

Copy images to test into models/object_detection/test_images directory. \
Renamed those pictures to be image3.jpg, image4.jpg...etc.

Go to models/research/object_detection and open jupyter notebook. \
Make some edits to change paths, etc. \
In the Variables section, change:
     # What model to download.
     MODEL_NAME = 'xxx_inference_graph'

     # Path to frozen detection graph. This is the actual model that is used for the object detection.
     PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

     # List of the strings that is used to add correct label for each box.
     PATH_TO_LABELS = os.path.join('training', 'xxx_label_map.pbtxt')

     NUM_CLASSES = 1
In the Detection section, change:
     TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(3, 8) ] \
Range should be however many pictures you put.

Run all object_detection_tutorial.ipynb!!!


References:
https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/
https://github.com/tzutalin/labelImg
https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
