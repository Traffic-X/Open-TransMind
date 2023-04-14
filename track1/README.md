English | [简体中文](README_ch.md)

## **Background**

### **Overview**

For well-designed network structures and loss functions, joint training of multiple tasks can greatly improve the generalization of the model.However, due to the noise in in specific task data, there is a risk of overfitting only using the data of a single task for training. The unified multi-task large model can average the noise of different tasks by integrating the data of multiple tasks for joint training,  thereby enabling the model to learn better features. To further explore the ability of  the unified multi-task large model, this track takes the typical task of traffic scenes as the topic, and combines the three CV tasks of classification, detection, and segmentation into a single large model. Ultimately, a single large model possesses the capabilities of the three CV tasks while achieving performance ahead of a specific single task model.

### **Introduction**

The previous mainstream visual model usually adopts a single-task "train from scratch" scheme. Each task is trained from scratch, and each task cannot learn from each other. Due to the bias of insufficient single-task data, the performance heavily depends on the distribution of task data, and the scene generalization is often poor. Recently, the booming large-scale data pre-training technology learns more general knowledge by using a large amount of data, and then migrates it to downstream tasks. The pre-training model obtained based on massive data has better knowledge completeness, and fine-tuning based on a small amount of data can still achieve better results in downstream tasks. However, based on the model production process of pre-training + downstream task fine-tuning, it is necessary to train models for each task separately, which consumes a lot of resources.

The VIMER-UFO ([*UFO：Unified Feature Optimization*](https://arxiv.org/pdf/2207.10341v1.pdf)) AllinOne multi-task training scheme proposed by Baidu can be directly applied to handle multiple tasks by using data from multiple tasks to train a powerful general-purpose model. The VIMER-UFO not only improves the performance of a single task through cross-task information, but also eliminates the fine-tuning process of downstream tasks. The VIMER-UFO AllinOne model can be widely used in various multi-task AI systems. Taking the smart city scene as an example, VIMER-UFO can use a single model to achieve the SOTA effect of multiple tasks such as face recognition, human body and vehicle ReID. At the same time, the multi-task model can achieve significantly better results than the single task model, demonstrating the effectiveness of the information reference mechanism between multiple tasks.

### **Task**

This track aims to improve the generalization ability of the model through multi-task joint training, and solves the conflict between different task. Based on traffic scenarios, this track selects three representative tasks of classification, detection, and segmentation for AllInOne joint training.

Task definition: Given the data set of the three tasks of classification, detection, and segmentation, a unified large model is used for AllInOne joint training, so that a single model has the ability of classification, detection, and segmentation.

#### **Dataset Introduction**

We used public datasets for classification, detection, and segmentation as follows:

##### **train dataset**

| Task                                               | Type           | Dataset               | Image num |
| -------------------------------------------------- | -------------- | --------------------- | --------- |
| Fine-Grained Image Classification on Stanford Cars | Classification | Stanford Cars         | 8,144     |
| Traffic Sign Recognition on Tsinghua-Tencent 100K  | Detection      | Tsinghua-Tencent 100K | 6,103     |
| Semantic Segmentation on BDD100K                   | Segmentation   | BDD100K               | 7,000     |

##### **test dataset**

| Task                                               | Type           | Dataset               | Image num |
| -------------------------------------------------- | -------------- | --------------------- | --------- |
| Fine-Grained Image Classification on Stanford Cars | Classification | Stanford Cars         | 8,041     |
| Traffic Sign Recognition on Tsinghua-Tencent 100K  | Detection      | Tsinghua-Tencent 100K | 3,067     |
| Semantic Segmentation on BDD100K                   | Segmentation   | BDD100K               | 1,000     |

#### **Data Description**

##### **Stanford Cars**

**Annotation format**：Each line consists of the image name and the corresponding category id

Reference annotation examples are shown below:

00001.jpg 0

00002.jpg 2

00003.jpg 1

...

##### **Tsinghua-Tencent 100K**

**Annotation format**：Refer to [*COCO annotation format*](https://cocodataset.org/#format-data)

##### **BDD100K**

**Annotation format**：Refer to [*BDD100K annotation format*](https://doc.bdd100k.com/download.html#semantic-segmentation)

### **Evaluation Metric**

Classification Task: Top-1 accuracy

Detection Task: mAP50

Segmentation Task: mIoU

Final score of A-list: The average of the evaluation results of the three tasks.

### **Competition Details**

The competition is divided into A/B lists. The A list is scored based on the prediction result files of the three tasks of classification, detection, and segmentation submitted by competitors. The B list requires competitors to submit codes and finetune scripts, and the results of the post-finetune test on the undisclosed data set will be scored.

Before the competition submission deadline, only the A list is visible to the competitors. After the competition, the B list will be announced to the competitors. The final ranking of the competition will be based on the combined ranking of the contestants' scores in the A-list and the B-list.

Note: Please ensure that the code finetune script submitted to the B list can run smoothly.

### **Submission Format**

File Format: JSON

Content Format:

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

Example:

https://aistudio.baidu.com/aistudio/datasetdetail/203253 (track1_submit_example.json)

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
