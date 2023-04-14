简体中文 | [English](README.md)

# 赛道二：Cross-Modal Image Retrieval Track
## 赛题背景
交通场景中高性能的图像检索能力对于交通执法、治安治理具有十分重要的作用，传统的图像检索方式通常使用先对图像进行属性识别再通过与期望属性的对比实现检索能力。随着多模态大模型技术的发展，文本与图像的表征统一和模态转换已有广泛应用，使用该能力可以进一步提升图像检索的精度和灵活性。
## 赛题任务
本赛道旨在提升交通场景中文本图像检索的精度。因此我们将多种公开数据集以及网络数据中的交通参与者图像进行了文本描述标注从而构建了多对多的图像-文本对，选手可以在此基础上进行多模态技术的研究工作，提升文本检索图像的精度。
### 数据集介绍
本赛题构建了一个多交通参与者的文本检索图像数据集，该数据集以开源数据集为基础，同时使用网络爬虫技术扩充数据的丰富度。在标注方面，首先利用CV大模型丰富图像标注属性，然后利用大语言模型构造图像对应的文本标注。目前数据集的总量有153728张，其中训练集136117张，评测集17611张。数据集包含行人和车辆2类交通参与者，数据分布具体见下表。
|  类别   | 训练集  | 测试集 |
|  ----  | ----  | ---- |
| 行人  | 90000 | 10000 |
| 车辆  | 46117 | 7611 |
| 总数  | 136117 | 17611 |
### 数据说明
标注格式示例：
```
图片$属性标注$文本
pmitevanhx.jpg$A white Audi.$This is a white Audi.
090000$0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0$A female pedestrian is someone who walks on foot, is between 18 and 60 years old, with her body facing the camera. She is in a short sleeve shirt with an upper logo.
```
由于存在一个文本对应多张图像 or 一张图像对应多个文本的情况，在评测过程中使用「属性文本」对这种情况进行处理。
## 评价指标
本赛题聚焦在文本检索图像的精度，因此使用的评价指标为平均检索精度$mAP@K$, $mAP$全称mean Average Precision，$K$代表的是前$K$个检索结果参与评价，本赛题取$K=10$，平均检索精度$mAP@K$的计算如下式所示：
$$mAP@K=\frac{1}{m} * \sum_{i=1}^{K}{p(i)*\Delta r(i)}，$$
其中，*m*为评测集中文本的总数，$p(i)$指的是$topi$检索结果的precision, $\Delta r(i)$的计算如下式所示：
$$\Delta r(i)=r(i)-r(i-1)，$$
其中，$r(i)$为$topi$结果的recall, $r(0)=0$。

## 提交格式
文件格式：JSON

内容格式：
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
**text_1, text2, ... 的顺序需要和test/test_text.txt中的顺序保持一致。**

参考示例：
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

## 比赛说明
该比赛分A/B榜单，A榜单基于选手提交预测结果进行打分和排序，B榜单针对topk选手提交的code进行复现，复现失败则排名无效。
比赛提交截止日期前仅A榜对选手可见，比赛结束后B榜会对选手公布，比赛最终排名按照选手成绩在A榜和B榜联和的排名。
注意：请确保提交至B榜上的代码脚本能够顺利运行。

参考示例：

https://aistudio.baidu.com/aistudio/datasetdetail/203278 (track2_submit_example.json)

# Track2 Codebase

## 使用方案

提供了文本图像检索模型训练方法
Demo使用单机（8卡）40G A100训练

### 环境配置

运行环境为python3.7，cuda11.0测试机器为A100。使用pip的安装依赖包，如下：
```bash
pip install -r requirements.txt
```

### 数据配置

从[官方数据下载地址](https://aistudio.baidu.com/aistudio/datasetdetail/203278/0)下载训练和测试数据后，将数据解压到datasets文件夹中（若不存在，请先创建）

### 训练

我们提供了vitbase的CLIP预训练权重，下载预训练权重至pretrained文件夹中（若不存在，请先创建），后使用以下脚本在训练集上启动训练

```bash
sh scripts/train.sh
```

### 生成预测json

```bash
sh scripts/infer.sh
```
