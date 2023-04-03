English | [简体中文](README_ch.md)

# Track1 Codebase

## Instructions

We provide a three task  AllInOne joint training method of classification, detection, and segmentation.

Demo is based on 8 A100 cards.

### Environment

Please use python3.7 and cuda11.0. 

```bash
pip install -r requirements.txt
```

### Data Configuration

After downloading the training and testing data from [official data download address](https://aistudio.baidu.com/aistudio/datasetdetail/203253), decompress the data into the 'datasets' folder (if it does not exist, please create it first)

### Training

We provide the pre-training weights on the object365 dataset, download the pre-training weights to the 'pretrained' folder (if it does not exist, please create it first), and then use the following script to start training

```bash
sh scripts/train.sh
```

### Inference

We provide the weights of our three-task AllinOne joint training, which can be downloaded to the 'pretrained' folder (if it does not exist, please create it first), and then use the following script to start inference

```bash
sh scripts/test.sh
```
