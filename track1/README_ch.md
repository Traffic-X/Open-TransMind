简体中文 | [English](README.md)

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
