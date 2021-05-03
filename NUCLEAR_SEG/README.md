
# Large Scale Solution to Nuclear Segmentation 
 A fully automatic framework to robostly segment Large Scale Nuclear Images 

##  1.Set up 
   ####  (Optional) Create virtual environment
    $ conda create -n BrainCellSeg python=3.6 anaconda
    $ conda activate BrainCellSeg
    
   ####  Installyation Requirements:  
   Python 3.6, **TensorFlow 2.1**,, Keras 2.0.8 and other common packages. Highly recommend to install the GPU verison of Tensorflow

   #### Option1 : CPU version
    $ pip install -r requirements_cpu.txt --user
    $ python3 setup.py install --user

   #### Option2 : GPU version
    $ module load cudatoolkit/10.1                      # confirmed good in Clusters: Sabine and Maui
    $ pip install -r requirements_gpu.txt --user
    $ cd mrcnn_Seg
    $ python3 setup.py install --user
    $ cd ..                                           # the project root folder
   
   #### Test demo
   Download the pretrained_weights.h5 to the project root from [GoogleDrive](https://drive.google.com/open?id=12algdsF7hxoF6lLepRoDed36gBx-NkCD)

   ```
   $ python3 main_segmentation.py detect \
    --dataset=demo/demo_input.jpeg \
    --weights="pretrained_weights.h5" \
    --toRGBOpt=1 \
    --results=demo\demo_out
   ```

##  2.Prepare Dataset       
 If there is only one nuclear stained channel (e.g. only DAPI), skip this step
- Input:
   -  `INPUT_DIR`: Datasets Path
   -  `DAPI`:Relative filename of DAPI image
   -  `HISTONES`:Relative filename of HISTONES image
- Output:
   - `OUTPUT_DIR`: Stacked Image for segmentation, located at /data 
   ```
    $ mkdir data
    $ python main_prepare_images.py \
    --INPUT_DIR=/path/to/input/dir \
    --OUTPUT_DIR=data \
    --DAPI R2C1.tif \
    --HISTONES R2C2.tif 
   ```

##  3. Test Segmentation with MRCNN 

- Input:
   - `weights`: Trained weight
   - `dataset`: Prepared Image or input gray image
- Output:
   - `results`: label image of whole brain segmentation , located at /results
   ```
   $ mkdir results
   $ python3 main_nucleiSeg.py detect \
    --dataset=data/multiplex.tif \
    --weights="pretrained_weights.h5" \
    --results=results
   ```
# Error shooting :

AttributeError: module 'tensorflow.python.keras.utils.generic_utils' has no attribute 'populate_dict_with_module_objects'
$ pip list | grep tf
$ pip install tensorflow --upgrade --force-reinstall




