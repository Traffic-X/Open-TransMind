简体中文 | [English](README.md)

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
