Mask RCNN

1. Make a conda environment
conda create -n MaskRCNN python=3.6 pip

source activate MaskRCNN

2. Install packages
pip install -r requirements.txt

3. Clone depository
git clone  https://github.com/matterport/Mask_RCNN.git

4. Install pycocotools
Note: pycocotools requires Visual C++ 2015 Build Tools (http://landinghub.visualstudio.com/visual-cpp-build-tools)

git clone https://github.com/philferriere/cocoapi.git

pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

5. Download the pre-trained weights from (https://github.com/matterport/Mask_RCNN/releases)

pip install imgaug
pip install opencv-python
jupyter notebook -> demo.ipynb (under sample file)
