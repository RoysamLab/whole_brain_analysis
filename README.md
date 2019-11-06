# WHOLE BRAIN ANALYSIS PIPELINE

## 1. Reconstruction
1. Specify your OS and select `main_reconstruction_lin.py` or `main_reconstruction_win.py`.

    __Notice__: You need __Matlab__ for windows.

2. Parse the arguments to  `main_reconstruction_*.py`:
  ```bash
  python main_reconstruction_lin.py \
       --INPUT_DIR /brazos/roysam/datasets/TBI/G2_Sham_Trained/G2_BR#22_HC_13L/original \
       --OUTPUT_DIR /brazos/roysam/datasets/TBI/G2_Sham_Trained/G2_BR#22_HC_13L \
       --MODE supervised \
       --SCRIPT scripts/20_plex.csv
  ```

  ```bash
  python main_reconstruction_lin.py \
       --INPUT_DIR /brazos/roysam/datasets/TBI/G2_Sham_Trained/G2_BR#22_HC_13L/original \
       --OUTPUT_DIR /brazos/roysam/datasets/TBI/G2_Sham_Trained/G2_BR#22_HC_13L \
       --MODE unsupervised \
       --DEFAULT_BOX 34000 8000 44000 15000 \
       --BRIGHTFIELD 11
  ```

  __Arguments:__
  - `INPUT_DIR` : Path to the directory containing input images
  - `OUTPUT_DIR` : Path to the directory saving output images
  - `MODE`: `supervised` or `unsupervised`
    - if `unsupervised`:
      - `DEFAULT_BOX` : `[xmin, ymin, xmax, ymax]`
      
      ![Alt text](files/default_box.png)
      
      - `BRIGHTFIELD` : Number of Brightfield channel or `None`
      
      Will generate a script in `OUTPUT_DIR/script.csv` like follows:
      
      filename |biomarker   |intra channel correction|inter channel correction|channel 1|channel 2|channel 3|xmin |ymin |xmax |ymax
      ---------|------------|------------------------|------------------------|---------|---------|---------|-----|-----|-----|-----
      R1C1.tif |            |Yes                     |Yes                     |         |         |         |34000|8000 |44000|15000
      R1C10.tif|            |Yes                     |Yes                     |R1C7.tif |R1C5.tif |         |34000|8000 |44000|15000
      R1C11.tif|Brightfield |No                      |No                      |         |         |         |34000|8000 |44000|15000
      R1C2.tif |            |Yes                     |Yes                     |R1C1.tif |         |         |34000|8000 |44000|15000
      R1C3.tif |            |Yes                     |Yes                     |         |         |         |34000|8000 |44000|15000
      R1C4.tif |            |Yes                     |Yes                     |R1C1.tif |         |         |34000|8000 |44000|15000
      R1C5.tif |            |Yes                     |Yes                     |R1C6.tif |         |         |34000|8000 |44000|15000
      R1C6.tif |            |Yes                     |Yes                     |R1C5.tif |         |         |34000|8000 |44000|15000
      R1C7.tif |            |Yes                     |Yes                     |R1C10.tif|R1C5.tif |         |34000|8000 |44000|15000
      R1C8.tif |            |Yes                     |Yes                     |R1C7.tif |         |         |34000|8000 |44000|15000
      R1C9.tif |            |Yes                     |Yes                     |         |         |         |34000|8000 |44000|15000
      R2C1.tif |            |Yes                     |Yes                     |         |         |         |34000|8000 |44000|15000
      R2C10.tif|            |Yes                     |Yes                     |R2C8.tif |         |         |34000|8000 |44000|15000
      R2C11.tif|Brightfield |No                      |No                      |         |         |         |34000|8000 |44000|15000
      R2C2.tif |            |Yes                     |Yes                     |         |         |         |34000|8000 |44000|15000
      R2C3.tif |            |Yes                     |Yes                     |R2C8.tif |R2C6.tif |         |34000|8000 |44000|15000
      R2C4.tif |            |Yes                     |Yes                     |         |         |         |34000|8000 |44000|15000
      R2C5.tif |            |Yes                     |Yes                     |R2C3.tif |R2C6.tif |         |34000|8000 |44000|15000
      R2C6.tif |            |Yes                     |Yes                     |R2C3.tif |R2C8.tif |         |34000|8000 |44000|15000
      R2C7.tif |            |Yes                     |Yes                     |R2C6.tif |R2C8.tif |R2C5.tif |34000|8000 |44000|15000
      R2C8.tif |            |Yes                     |Yes                     |         |         |         |34000|8000 |44000|15000
      R2C9.tif |            |Yes                     |Yes                     |         |         |         |34000|8000 |44000|15000

    - if `supervised` :
      - `SCRIPT` : Path to the updated `script.csv` file for reconstruction configuration:
      
      filename |biomarker   |intra channel correction|inter channel correction|channel 1|channel 2|channel 3|xmin |ymin |xmax |ymax
      ---------|------------|------------------------|------------------------|---------|---------|---------|-----|-----|-----|-----
      R1C1.tif |DAPI        |Yes                     |No                      |         |         |         |34000|8000 |44000|15000
      R1C10.tif|NFH         |No                      |Yes                     |R1C7.tif |         |         |34000|8000 |44000|15000
      R1C11.tif|Brightfield |No                      |No                      |         |         |         |34000|8000 |44000|15000
      R1C2.tif |DAPI        |Yes                     |No                      |         |         |         |34000|8000 |44000|15000
      R1C3.tif |CC3         |Yes                     |No                      |         |         |         |34000|8000 |44000|15000
      R1C4.tif |NeuN        |Yes                     |No                      |         |         |         |34000|8000 |44000|15000
      R1C5.tif |MBP         |No                      |Yes                     |R1C6.tif |         |         |34000|8000 |44000|15000
      R1C6.tif |RECA1       |Yes                     |Yes                     |R1C5.tif |R1C8.tif |         |22000|24000|32000|31000
      R1C7.tif |IBA1        |Yes                     |Yes                     |R1C10.tif|R1C5.tif |R1C8.tif |34000|8000 |44000|15000
      R1C8.tif |TomatoLectin|Yes                     |Yes                     |R1C7.tif |         |         |34000|8000 |44000|15000
      R1C9.tif |PCNA        |Yes                     |No                      |         |         |         |34000|8000 |44000|15000
      R2C1.tif |DAPI        |Yes                     |No                      |         |         |         |34000|8000 |44000|15000
      R2C10.tif|MAP2        |No                      |Yes                     |R2C8.tif |         |         |34000|8000 |44000|15000
      R2C11.tif|Brightfield |No                      |No                      |         |         |         |34000|8000 |44000|15000
      R2C2.tif |DAPI        |Yes                     |No                      |         |         |         |34000|8000 |44000|15000
      R2C3.tif |GAD67       |Yes                     |Yes                     |R2C8.tif |R2C6.tif |         |34000|16000|44000|23000
      R2C4.tif |GFAP        |Yes                     |No                      |         |         |         |34000|8000 |44000|15000
      R2C5.tif |Parvalbumin |Yes                     |Yes                     |R1C5.tif |R2C6.tif |         |31000|12000|41000|19000
      R2C6.tif |S100        |Yes                     |Yes                     |R2C3.tif |R2C8.tif |         |34000|8000 |44000|15000
      R2C7.tif |Calretinin  |Yes                     |Yes                     |R2C6.tif |R2C8.tif |         |18000|1000 |28000|17000
      R2C8.tif |TomatoLectin|Yes                     |No                      |         |         |         |34000|8000 |44000|15000
      R2C9.tif |CD31        |Yes                     |No                      |         |         |         |34000|8000 |44000|15000
 
## 2. Detection
1. __setup__:
    - Download executable from [here](https://github.com/google/protobuf/releases) and run the following command from
     ```lib``` directory [(read more)](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#protobuf-compilation):
    ``` bash
    # from DETECTION/lib
    protoc object_detection/protos/*.proto --python_out=.
    ```
2. Parse the arguments to  `main_detection.py`:
  - __if only DAPI:__
    ```bash
    python main_detection.py \
       --INPUT_DIR=/brazos/roysam/datasets/TBI/G2_Sham_Trained/G2_BR#22_HC_13L/original \
       --OUTPUT_DIR=/brazos/roysam/datasets/TBI/G2_Sham_Trained/G2_BR#22_HC_13L/detection_results \
       --DAPI=R2C1.tif
    ```
  - __if DAPI + Histones:__
    ```bash
    python main_detection.py \
       --INPUT_DIR=/brazos/roysam/datasets/TBI/G2_Sham_Trained/G2_BR#22_HC_13L/original \
       --OUTPUT_DIR=/brazos/roysam/datasets/TBI/G2_Sham_Trained/G2_BR#22_HC_13L/detection_results \
       --DAPI=R2C1.tif \
       --HISTONES=R2C2.tif
    ```
