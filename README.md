# Word2Vec 教学教程

本项目提供了一个简明易懂的 Word2Vec 模型实现与教程，帮助大家理解词嵌入(Word Embedding)的基本原理和实现方法。

## 项目概述

Word2Vec 是一种常用的词嵌入技术，能够将文本中的词语映射到低维向量空间中，捕捉词语之间的语义关系。本教程通过简单的中文文本实例，使用 PyTorch 实现了一个基础的 Word2Vec 模型，并展示了词向量的可视化和应用。

## 功能特点

- 使用 PyTorch 从零实现简单的 Word2Vec 模型
- 支持中文文本处理（使用 jieba 分词）
- 包含词向量的训练、可视化和相似度计算
- 详细的步骤说明和代码注释，便于学习理解

## 环境要求

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- jieba (中文分词库)

## 使用说明

1. 克隆本仓库或下载教程文件
2. 安装所需依赖：`pip install torch numpy matplotlib scikit-learn jieba`
3. 打开`word2vec_tutorial.ipynb`文件，按步骤运行即可

## 教程内容

教程包含以下主要部分：

1. **基础准备**：导入必要的库和设置参数
2. **数据处理**：准备示例文本和分词处理
3. **构建词汇表**：创建词到索引的映射
4. **准备训练数据**：生成训练样本（中心词-上下文词对）
5. **模型定义**：实现简单的 Word2Vec 模型
6. **模型训练**：训练词向量模型并输出损失变化
7. **词向量可视化**：使用 t-SNE 可视化词向量空间
8. **词向量应用**：计算词语相似度和查找相似词

## 示例效果

教程展示了如何训练词向量模型，并通过可视化展示词语之间的语义关系。例如，可以查找与"中国"、"北京"、"人工智能"等词语相似的词，展示了模型对语义关系的捕捉能力。

## 进一步学习

- 尝试使用更大的语料库训练更好的词向量
- 调整窗口大小和嵌入维度等参数，观察对结果的影响
- 尝试实现 Skip-gram 和 CBOW 两种 Word2Vec 模型
- 将训练好的词向量用于下游 NLP 任务，如文本分类、情感分析等

## 参考资料

- Mikolov, T., et al. (2013). [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
