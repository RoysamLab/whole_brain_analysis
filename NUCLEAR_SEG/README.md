

    
# Large Scale Solution to Nuclear Segmentation with No Annotations 
 A fully automatic framework to robostly segment Large Scale Nuclear Images 

##  1.Set up 
   ####  Create virtual environment
    $ conda create -n seg python=3.6 anaconda
    $ conda activate seg
    $ cd SegmentationPipeline

   ####  Installyation Requirements:  
   Python 3.6, TensorFlow 1.3, Keras 2.0.8 and other common packages. Highly recommend to install the GPU verison of Tensorflow
   $ cd mrcnn_Seg

   #### Option1 : CPU version
    $ pip install -r requirements_cpu.txt --user
   #### Option2 : GPU version
    $ module load cudatoolkit/10.1                      # open a GPU node to install
    $ pip install -r requirements_gpu.txt --user
    $ cd mrcnn_Seg

    
   $ python3 setup.py install --user
   $ cd ..                                          # the project root folder

    Tips: check whether GPU is running : $ watch -n 2 nvidia-smi

   #### Test demo
    $ python3 main_segmentation.py detect \
        --dataset=demo/demo_input.jpeg \
        --weights="pretrained_weights.h5" \
        --results=demo

##  3. Initial Segmentation       
 #### Create noisy training labels 
 Use traditional segmentation method to generate the rough segmentation result
     $ cd Automatic_Seg
     $ python main_autoSeg_tiled_script.py \
        --method="watershed"
        --dataset="/project/hnguyen/xiaoyang/exps/Data/50_plex/jj_final/images_stacked/DPH.tif" \
        --results="/project/hnguyen/xiaoyang/exps/Data/50_plex/out/autoseg"
        
     $ python main_autoSeg_tiled_script.py \
--method="yosef" \
--dataset="/project/hnguyen/xiaoyang/exps/Data/50_plex/jj_final/images_stacked/DPH.tif" \
--results="/project/hnguyen/xiaoyang/exps/Data/50_plex/out/autoseg"
        
  __Arguments:__
  - `dataset` : Path the input gray scale image
  - `results` : Path to the directory saving output segmentation labels
  - `method`: `watershed` or `yosef` segmentation methods


`

##  Step4： Prepare training set
### Stack 8 channels multiplex image 

> - Output
>   -  "$data_dir"/images_stacked_multiplex/multiplex.tif

     $ python3 "$data_dir"/images_stacked_multiplex/Prepare_Image_Stacked_Multiplex.py

### Crop and mold the training set 

>- Input:
>   -  {img} multiple Image  
>   -  {label} autoseg result label image
>- Output:
>   -  {results} Cropped image and binary label images  
	>   e.g. **/project/roysam/xli63/exps/SegmentationPipeline/mrcnn_Seg/datasets/50_plex/multiplex/whole**

    $ img_dir="$data_dir"/images_stacked_multiplex/multiplex.tif 
    $ label_dir="$data_dir"/autoseg/DPH/[DAPI+Histones]204215seeds_fill_Falselabels_nuclear.txt 
    $ dataset_root="$proj_root"/mrcnn_Seg/datasets/50_plex/RDGHBDPH/whole

    $ cd "$proj_root"/mrcnn_Seg/supplement
    $ python3 DataPrepare_mrcnn.py train \
        --img="$img_dir" \
        --label="$label_dir" \
        --results="$dataset_root" \
        --multiprocess=1 \
        --save_display=1 \
        --masks_fName=masks_autoseg \
        --imadjust=0

##  Step5：Train Segmentation with MRCNN 
>- Input:
>   -  {dataset} Cropped image and binary label images  
>   - {weights} Pretrain weight
>   - {masks_fName} annotation from autoseg:"masks_autoseg"
>- Output:
>   -  {logs}: Trained weight
>   e.g. **/project/roysam/xli63/exps/SegmentationPipeline/mrcnn_Seg/results/[train]-RDGHBDPH-molded_autoseg-imagenet/nucleus20190423T1044/mask_rcnn_nucleus_0040.h5**

    $ fName=RDGHBDPH-molded_autoseg-imagenet
    $ result_dir="$proj_root"/results/[train]-"$fName"
    $ mkdir "$result_dir"

    $ cd "$proj_root"/samples/Hippo
    $ python3 nucleus_wholebrain_train_detect_merge_multiplex.py train \
    --dataset="$dataset_root"/whole/train \
    --weights="imagenet" \
    --toRGBOpt=1 \
    --masks_fName="masks_autoseg" \
    --logs="$result_dir" \
    2>&1 | tee "$result_dir"/log.txt
    
After training (taked few hrs), plot the training loss

    $ python "$proj_root"/supplement/PlotTrainingLog.py \
    -l "$result_dir"/log.txt \
    -r "$result_dir"
Save the pretrain weights

    $ cd "$result_dir"/nucleus*
    $ target_weight_loc="$proj_root"/samples/Hippo/weights/mask_rcnn_nucleus_0040[train]-"$fName".h5
    $ scp mask_rcnn_nucleus_0040.h5 "$target_weight_loc"
    $ echo $target_weight_loc  


##  Step6：Test Segmentation with MRCNN 
>- Input:
>   -  {weights}Trained weight
>   -  {dataset}Raw image 
>- Output:
>   -  {results}Updated label image for whole brain
	>   **e.g./project/roysam/xli63/exps/Data/50_plex/jj_final/seg_results/RDGHBDPH-molded_autoseg-imagenet**

    $ dataset_dir="$data_root"/images_stacked
    $ cd "$proj_root"/samples/Hippo
    $ result_dir="$data_root"/seg_results/"$fName"

    $ python3 nucleus_wholebrain_train_detect_merge_multiplex.py detect \
    --dataset="$dataset_dir"/RDGHBDPH.tif \
    --weights=weights/mask_rcnn_nucleus_0040"$fName".h5 \
    --results="$result_dir" \
    --imadjust=F \
    2>&1 | tee "$result_dir"/log[test].txt



# Error shooting :

AttributeError: module 'tensorflow.python.keras.utils.generic_utils' has no attribute 'populate_dict_with_module_objects'
$ pip list | grep tf
$ pip install tensorflow --upgrade --force-reinstall
