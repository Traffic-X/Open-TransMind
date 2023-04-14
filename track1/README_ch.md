简体中文 | [English](README.md)

## **赛题背景**

### **整体概述**

对于设计精良的网络结构和损失函数，多个任务共同训练能大幅提升模型的泛化性。由于特定任务的数据存在noise，仅使用单一任务的数据进行训练，存在过拟合的风险。统一多任务大模型通过将多个任务的数据整合进行统一训练，能够对不同任务的noise做一个平均，进而使模型学到更好的特征。为了进一步探索统一多任务大模型的能力上限，本赛道以交通场景典型任务为题，联合分类、检测、分割三项CV任务三大数据集至单一大模型中，使得单一大模型具备能力的同时获得领先于特定单任务模型的性能。

### **原理介绍**

之前主流的视觉模型生产流程，通常采用单任务 “train from scratch” 方案。每个任务都从零开始训练，各个任务之间也无法相互借鉴。由于单任务数据不足带来偏置问题，实际效果过分依赖任务数据分布，场景泛化效果往往不佳。近两年蓬勃发展的大数据预训练技术，通过使用大量数据学到更多的通用知识，然后迁移到下游任务当中，本质上是不同任务之间相互借鉴了各自学到的知识。基于海量数据获得的预训练模型具有较好的知识完备性，在下游任务中基于少量数据 fine-tuning 依然可以获得较好的效果。不过基于预训练+下游任务 fine-tuning 的模型生产流程，需要针对各个任务分别训练模型，存在较大的研发资源消耗。

百度提出的 VIMER-UFO（[*UFO：Unified Feature Optimization*](https://arxiv.org/pdf/2207.10341v1.pdf)） All in One 多任务训练方案，通过使用多个任务的数据训练一个功能强大的通用模型，可被直接应用于处理多个任务。不仅通过跨任务的信息提升了单个任务的效果，并且免去了下游任务 fine-tuning 过程。VIMER-UFO All in One 研发模式可被广泛应用于各类多任务 AI 系统，以智慧城市场景为例，VIMER-UFO 可以用单模型实现人脸识别、人体和车辆ReID等多个任务的 SOTA 效果，同时多任务模型可获得显著优于单任务模型的效果，证明了多任务之间信息借鉴机制的有效性。

### **赛题任务**

本赛道旨在通过多任务联合训练来提升模型的泛化能力，同时解决多任务、多数据之间冲突的问题。本赛题基于交通场景，选择了分类、检测、分割三大代表性任务进行AllInOne联合训练。

任务定义：根据给出的分类、检测、分割三任务的数据集，使用统一大模型进行AllInOne联合训练，使得单一模型能够具备分类、检测、分割的能力。

#### **数据集介绍**

我们使用了分类、检测、分割的公开数据集具体如下:

##### **训练集**

| 任务                                               | 任务类别 | 数据集                | 图片数 |
| -------------------------------------------------- | -------- | --------------------- | ------ |
| Fine-Grained Image Classification on Stanford Cars | 分类     | Stanford Cars         | 8,144  |
| Traffic Sign Recognition on Tsinghua-Tencent 100K  | 检测     | Tsinghua-Tencent 100K | 6,103  |
| Semantic Segmentation on BDD100K                   | 分割     | BDD100K               | 7,000  |

##### **测试集**

| 任务                                               | 任务类别 | 数据集                | 图片数 |
|----------------------------------------------------|----------|-----------------------|--------|
| Fine-Grained Image Classification on Stanford Cars | 分类     | Stanford Cars         | 8,041  |
| Traffic Sign Recognition on Tsinghua-Tencent 100K  | 检测     | Tsinghua-Tencent 100K | 3,067  |
| Semantic Segmentation on BDD100K                   | 分割     | BDD100K               | 1,000  |

#### **数据说明**

##### **Stanford Cars**

**标注格式**：每行由图片名称和对应类别id组成

参考标注实例如下方所示：

00001.jpg 0

00002.jpg 2

00003.jpg 1

...

##### **Tsinghua-Tencent 100K**

**标注格式**：参考[*COCO标注格式*](https://cocodataset.org/#format-data)

##### **BDD100K**

**标注格式**：参考[*BDD100K标注格式*](https://doc.bdd100k.com/download.html#semantic-segmentation)

### **评价指标**

分类任务：Top-1 accuracy

检测任务：mAP50

分割任务：mIoU

A榜最终得分：三任务指标平均值

### **比赛说明**

比赛分A/B榜单，A榜基于选手提交的分类、检测、分割三任务预测结果文件进行打分；B榜需要选手提交代码和一键启动finetune脚本，在未公开数据集上进行finetune后测试的结果进行打分。

比赛提交截止日期前仅A榜对选手可见，比赛结束后B榜会对选手公布，比赛最终排名按照选手成绩在A榜和B榜联和的排名。

注意：请确保提交至B榜上的代码finetune脚本能够顺利运行

### **提交格式**

文件格式：JSON

内容格式：

{

​	'cls': {

​		'image_name': pred_cls_id, 

​		... ...

​	},

​	'dec': [

​		{'image_id': 1, 'category_id': 37, "bbox": [x, y, w, h], "score": pred_score},

​		... ...

​	],

​	'seg': {

​		'image_name': {'pred_cls_id': pred_polygons, ... ...},

​		... ...	

​	}

}

参考示例：

https://aistudio.baidu.com/aistudio/datasetdetail/203253 (track1_submit_example.json)


# Track1 Codebase

## 使用方案

提供了分类、检测、分割AllInOne三任务联合训练方法
Demo为单机（8卡）40G A100的训练方法

### 环境配置

运行环境为python3.7，cuda11.0测试机器为A100。使用pip的安装依赖包，如下：
```bash
pip install -r requirements.txt
```

### 数据配置

从[官方数据下载地址](https://aistudio.baidu.com/aistudio/datasetdetail/203253)下载训练和测试数据后，将数据解压到datasets文件夹中（若不存在，请先创建）

### 训练

我们提供了object365数据集的预训练权重，下载预训练权重至pretrained文件夹中（若不存在，请先创建），后使用以下脚本在训练集上启动训练

```bash
sh scripts/train.sh
```

### 预测

我们提供了我们训练的三任务AllinOne联合训练的权重，可下载权重至pretrained文件夹中（若不存在，请先创建），后使用以下脚本在测试集上启动预测

```bash
sh scripts/test.sh
```
