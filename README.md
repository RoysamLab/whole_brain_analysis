# WHOLE BRAIN ANALYSIS PIPELINE

## 1. Image Reconstruction

## 2. Detection
1. __setup__:
    - Download executable from [here](https://github.com/google/protobuf/releases) and run the following command from
     ```lib``` directory:
    ``` bash
    # from 2_DETECTION/lib
    protoc object_detection/protos/*.proto --python_out=.
    ```