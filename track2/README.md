English | [简体中文](README_ch.md)

# Track 2: Cross-Modal Image Retrieval Track

## Background
High-performance image retrieval in traffic scenes plays a crucial role in traffic law enforcement and public security management. Traditional image retrieval methods usually use attribute recognition to retrieve images by comparing with the expected attributes. With the development of multi-modal large model technology, the unification of text and image representation and modal conversion has been widely used. Using this ability can further improve the accuracy and flexibility of image retrieval.

## Task
The goal of this track is to improve the accuracy of text-based image retrieval in traffic scenes. Therefore, we have annotated text descriptions for images of traffic participants from various public datasets and online sources to construct many-to-many image-text pairs. Participants can conduct research on multimodal techniques based on these pairs to improve the accuracy of text retrieval for images.

### Dataset Introduction
This competition constructed a text retrieval image dataset with multiple traffic participants, based on open-source datasets and using web crawler technology to enrich the data. In terms of annotation, first, a large CV model is used to enrich the image annotation attributes, and then a large language model is used to construct the corresponding text annotation for the image. Currently, the dataset has a total of 153,728 images, including 136,117 images in the training set and 17,611 images in the evaluation set. The dataset contains two categories of traffic participants: pedestrians and vehicles, and the specific data distribution is shown in the table below.
|  Category   | Training Set | Test Set	 |
|  ----  | ----  | ---- |
| 行人  | 90000 | 10000 |
| 车辆  | 46117 | 7611 |
| 总数  | 136117 | 17611 |

### Data Description
Annotation format example:
```
Image$Attribute Annotation$Text
pmitevanhx.jpg$A white Audi.$This is a white Audi.
090000$0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0$A female pedestrian is someone who walks on foot, is between 18 and 60 years old, with her body facing the camera. She is in a short sleeve shirt with an upper logo.
```
Due to the existence of cases where one text corresponds to multiple images or one image corresponds to multiple texts, during the evaluation process, the "attribute text" is used to handle such cases.

## Evaluation Metric
This competition focuses on the accuracy of text retrieval for images, so the evaluation metric used is the mean Average Precision ($mAP@K$). Here, $K$ represents the number of retrieval results ($top K$) used in the evaluation, and this competition takes $K=10$. The calculation of $mAP@K$ is as follows:
$$mAP@K=\frac{1}{m} * \sum_{i=1}^{K}{p(i)*\Delta r(i)}，$$
where $m$ is the total number of texts in the evaluation set, $p(i)$ is the precision of the $topi$ retrieval results, and $\Delta r(i)$ is calculated as follows:
$$\Delta r(i)=r(i)-r(i-1)，$$
where $r(i)$ is the recall of the topi result, and $r(0)=0$.

## Submission Format
File format: JSON

Content format:
```
{
    {
        'results':
        [
            {
                'text': text_1,
                'image_names': [image_name_1, image_name_2, ..., image_name_10]
            },
            {
                'text': text_2,
                'image_names': [image_name_1, image_name_2, ..., image_name_10]
            },
            ...
        ]
    }
}
```

**The order of text_1, text2, ... must be consistent with the order in test/test_text.txt.**

Example:
```
{
    {
        'results':
        [
            {
                'text': 'This is a grey Chery Sedan.',
                'image_names': ['vehicle_0001886.jpg', 'vehicle_0000196.jpg', 'vehicle_0002886.jpg', 'vehicle_0007116.jpg', 'vehicle_0001256.jpg', 'vehicle_0007852.jpg', 'vehicle_0003548.jpg', 'vehicle_0008215.jpg', 'vehicle_0000851.jpg', 'vehicle_0007531.jpg']
            },
            {
                'text': 'This is a black Volkswagen Sedan.',
                'image_names': ['vehicle_0003234.jpg', 'vehicle_0009561.jpg', 'vehicle_0008521.jpg', 'vehicle_0006540.jpg', 'vehicle_0000851.jpg', 'vehicle_0003612.jpg', 'vehicle_0008124.jpg', 'vehicle_0000513.jpg', 'vehicle_0004811.jpg', 'vehicle_0001577.jpg']
            },
            ...
        ]
    }
}
```

## Competition Details
This competition is divided into A/B lists. The A list is based on the prediction results submitted by the participants for scoring and ranking, while the B list is for reproducing the code submitted by the top K participants. If the reproduction fails, the ranking will be invalid.

Before the competition submission deadline, only the A list is visible to the participants. After the competition ends, the B list will be made public to the participants. The final ranking of the competition is based on the participants' scores in the A and B lists combined.

Note: Please make sure that the code script submitted to the B list can run smoothly.

Example:

https://aistudio.baidu.com/aistudio/datasetdetail/203278 (track2_submit_example.json)

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
