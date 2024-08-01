# [KBS 2024] ACT
This repo is the official implementation of [Asymmetric Co-Training with Explainable Cell Graph Ensembling for Histopathological Image Classification]() which is conditionally accepted at Knowledge-Based Systems (KBS).

## Requirements
Some important required packages include:
* python==3.6 
* torch==1.7.1
* torch-geometric==1.7.2
* torch-cluster==1.5.9
* torch-scatter==2.0.6
* torch-sparse==0.6.9
* efficientnet-pytorch==0.7.1
* ......

To reproduce the paper's environment, please use the command: 
```shell
pip install -r requirements.txt
```

## Usage
### Clone the repo:
```
git clone https://github.com/NeuronXJTU/ACT.git
cd ACT
```
### Download the processed data:
🔥🔥🔥 The **preprocessed public [CRC dataset]() and private [LUAD7C dataset]()** are available for downloading. Please contact yangziqi@stu.xjtu.edu.cn or zhongyuli@xjtu.edu.cn for data.

### Cell Segmentation
#### For the public CRC dataset
* Place the CRC image patches to be segmented in the path `data/patches/CRC_datasets/`
* Use the cell segmentation network to perform cell segmentation on the data
* Place the generated mask images after segmentation in the path `data/patches_masks/CRC_datasets/` 
* The folder should be organized as follows:
```shell
./data/
├── patches
│   ├── CRC_datasets
│   │   ├── Patient_001_01_Low-Grade_0_0.png
│   │   ├── ...
├── patches_masks
│   ├── CRC_datasets
│   │   ├── Patient_001_01_Low-Grade_0_0.png
│   │   ├── ...
```
#### For our private LUAD7C dataset
* Place the LUAD image patches to be segmented in the path `data/LUNG_256_random/{i}/image`, here, `i` refers to the LUAD category
* Use the cell segmentation network to perform cell segmentation on the data
* Place the generated mask images after segmentation in the path `data/LUNG_256_random/{i}/mask`, here, `i` refers to the LUAD category
* The folder should be organized as follows:
```shell
./data/LUAD7C/
├── 1
│   ├── image
│   └── mask
├── 2
│   ├── image
│   └── mask
├── 3
│   ├── image
│   └── mask
├── 4
│   ├── image
│   └── mask
├── 5
│   ├── image
│   └── mask
├── 6
│   ├── image
│   └── mask
└── 12
    ├── image
    └── mask
```


### Cell Graph Construction

1. Run `python run_graph_construction.py`, fill in `data_name='name of the dataset to be processed'` in `run_graph_construction.py`, the generated cell graphs will be stored in pt format

2. Refer to the `get_txt` function in `run_graph_construction.py` to split the generated cell graph data into training and testing sets

### Asymmetric Co-Training

* Use the script `bash test/train.sh` to call `train.py` and start training. By default, `num_net=2`, indicating asymmetric co-training of Deep GCN and CNN

* The implemented `train.py` currently supports asymmetric co-training of 3 different network architectures, including GCN, CNN, and ViG

* Network architectures already implemented in `architecture.py` include:
```python
GCN_LIST = ['ResGCN14', 'DenseGCN14', 'ResGCN14_0', 'ResGCN14_1', 'DenseGCN14_0', 'DenseGCN14_1', 'PlainGCN', 'CGCNet']
CNN_LIST = ['ResNet18', 'ResNet18_0', 'ResNet18_1', 'ResNet34', 'ResNet50', 'DenseNet121', 'MobileNetV2',
            'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4']
ViG_LIST = ['pvig_ti_224_gelu', 'pvig_s_224_gelu', 'pvig_m_224_gelu', 'pvig_b_224_gelu']
```
