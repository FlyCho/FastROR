# Fast_ROR
### Introduction
The end-to-end framework for blister identification task
### Install
+ Python3.6
+ tensorflow 1.12.0
+ openCV
```
pip install -r requirements.txt
git clone -b dev https://github.com/FlyCho/FastROR.git
```
### Dataset fromat
```
The data is VOC format, reference [here](sample.xml)       
Data path format  ($R2CNN_ROOT/data/io/divide_data.py)    
```
├── datasets
│   ├── train
│       ├── Annotations
│       ├── JPEGImages
│    ├── test
│       ├── Annotations
│       ├── JPEGImages
``` 
```
### Generate the geometric and score map
```
python score_geo_map_prepare.py --dataset_dir=/path/to/your/training/set
```
### Two steps training strategy
We firstly train the localization network and then train the whole network, including the localization and recognition network.
### First step : training the localization network
```
python localization_train.py --gpu_list='0' --learning_rate=0.0001 --train_stage=2 --training_data_dir=/path/to/your/training images/ --training_gt_data_dir=/path/to/your/training annotations/
```
### Second step : training the whole network
```
python loc_recog_train.py --gpu_list='0' --learning_rate=0.0001 --train_stage=2 --training_data_dir=/path/to/your/training images/ --training_gt_data_dir=/path/to/your/training annotations/
```
### Test
```
python loc_recog_test.py --gpu_list='0' --test_data_path='/path/to/your/testing images/' --test_gt_path='/path/to/your/testing annotations/' --checkpoint_path='checkpoints/'
```
### Demo
```
python loc_recog_demo.py
```
### Examples
![image_1](demo_image/identify_result.jpg)

### Reference
+ [FOTS_TF](https://github.com/Pay20Y/FOTS_TF/tree/dev)
Thanks for the authors!
