English | [简体中文](README_ch.md)

# Track2 Codebase

## Instructions

We provide a training code for text-image retrieval task.

Demo is based on 8 A100 cards.

### Environment

Please use python3.7 and cuda11.0. 

```bash
pip install -r requirements.txt
```

### Data Configuration

After downloading the training and testing data from [official data download address](https://aistudio.baidu.com/aistudio/datasetdetail/203278/0), decompress the data into the 'datasets' folder (if it does not exist, please create it first)

### Training

We provide the vitbase pre-training weights of CLIP, download the pre-training weights to the 'pretrained' folder (if it does not exist, please create it first), and then use the following script to start training

```bash
sh scripts/train.sh
```

### Generate JSON for predictions

```bash
sh scripts/infer.sh
```
