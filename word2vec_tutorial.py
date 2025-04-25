#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Word2Vec 教学教程
本教程将带你了解Word2Vec的基本原理和实现。
我们将使用PyTorch实现一个简单的Word2Vec模型，
并通过可视化来理解词向量的含义。
"""

# 1. 导入必要的库
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import time
import jieba
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random

# 设置随机种子，确保结果可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 2. 准备示例数据
# 我们使用一些简单的中文句子作为示例数据
sentences = [
    "自然语言处理是人工智能的一个重要分支",
    "词嵌入是自然语言处理中的基础技术",
    "word2vec是一种常用的词嵌入模型",
    "词嵌入可以捕捉词语之间的语义关系",
    "中国是一个有着悠久历史的国家",
    "北京是中国的首都",
    "上海是中国的经济中心",
    "人工智能技术在各个领域得到了广泛应用",
    "深度学习是机器学习的一个重要分支",
    "自然语言处理技术可以帮助计算机理解人类语言"
] * 5  # 重复多次以增加样本量

# 展示分词效果
print("分词示例：")
for sentence in sentences[:2]:
    words = list(jieba.cut(sentence))
    print(f"原句：{sentence}")
    print(f"分词：{' '.join(words)}")
    print()

# 3. 构建词汇表
def build_vocab(sentences):
    """构建词汇表"""
    word_counts = Counter()
    for sentence in sentences:
        # 使用jieba进行中文分词
        words = list(jieba.cut(sentence))
        word_counts.update(words)
    
    # 创建词到索引的映射
    word_to_idx = {word: i for i, word in enumerate(word_counts.keys())}
    idx_to_word = {i: word for word, i in word_to_idx.items()}
    
    return word_to_idx, idx_to_word, len(word_counts)

# 构建词汇表
word_to_idx, idx_to_word, vocab_size = build_vocab(sentences)

# 展示词汇表信息
print(f"词汇表大小: {vocab_size}")
print("\n部分词汇示例：")
for word, idx in list(word_to_idx.items())[:10]:
    print(f"{word}: {idx}")

# 4. 准备训练数据
def prepare_data(sentences, word_to_idx, window_size=2):
    """准备训练数据"""
    data = []
    for sentence in sentences:
        # 使用jieba进行中文分词
        words = list(jieba.cut(sentence))
        for i, word in enumerate(words):
            if word not in word_to_idx:
                continue
            # 获取上下文词
            for j in range(max(0, i-window_size), min(len(words), i+window_size+1)):
                if i != j and words[j] in word_to_idx:
                    data.append((word_to_idx[word], word_to_idx[words[j]]))
    return data

# 准备训练数据
data = prepare_data(sentences, word_to_idx)

# 展示训练样本
print(f"训练样本数: {len(data)}")
print("\n部分训练样本示例：")
for center_idx, context_idx in data[:5]:
    print(f"中心词: {idx_to_word[center_idx]}, 上下文词: {idx_to_word[context_idx]}")

# 5. 定义Word2Vec模型
class SimpleWord2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SimpleWord2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        embeds = self.embeddings(x)
        out = self.linear(embeds)
        return out

# 创建模型
embedding_dim = 50
model = SimpleWord2Vec(vocab_size, embedding_dim)

# 打印模型结构
print(model)

# 6. 训练模型
def train_word2vec(sentences, embedding_dim=50, window_size=2, epochs=10, learning_rate=0.01):
    """训练Word2Vec模型"""
    start_time = time.time()
    
    # 构建词汇表
    word_to_idx, idx_to_word, vocab_size = build_vocab(sentences)
    print(f"词汇表大小: {vocab_size}")
    
    # 准备数据
    data = prepare_data(sentences, word_to_idx, window_size)
    print(f"训练样本数: {len(data)}")
    
    # 创建模型
    model = SimpleWord2Vec(vocab_size, embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # 训练
    print("开始训练...")
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for center, context in data:
            # 准备输入和目标
            center_tensor = torch.LongTensor([center])
            context_tensor = torch.LongTensor([context])
            
            # 前向传播
            output = model(center_tensor)
            loss = criterion(output, context_tensor)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss/len(data)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    end_time = time.time()
    print(f"训练完成，耗时: {end_time - start_time:.2f}秒")
    
    return model, word_to_idx, idx_to_word, losses

# 训练模型
model, word_to_idx, idx_to_word, losses = train_word2vec(
    sentences,
    embedding_dim=50,
    window_size=2,
    epochs=10,
    learning_rate=0.01
)

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 7. 词向量可视化
def visualize_word_vectors(model, word_to_idx, words_to_visualize=None):
    """可视化词向量"""
    if words_to_visualize is None:
        words_to_visualize = list(word_to_idx.keys())[:50]  # 默认可视化前50个词
    
    # 获取词向量
    vectors = []
    words = []
    for word in words_to_visualize:
        if word in word_to_idx:
            with torch.no_grad():
                vector = model.embeddings(torch.LongTensor([word_to_idx[word]])).numpy()[0]
            vectors.append(vector)
            words.append(word)
    
    # 将列表转换为NumPy数组
    vectors = np.array(vectors)
    
    # 使用t-SNE降维
    perplexity_value = min(30, max(1, len(vectors)-1))  # 确保perplexity小于样本数
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    vectors_2d = tsne.fit_transform(vectors)
    
    # 配置matplotlib支持中文显示 - 跨平台解决方案
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制散点图
    plt.figure(figsize=(15, 15))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
    
    # 添加词标签
    for i, word in enumerate(words):
        plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=9)
    
    plt.title('Word Vectors Visualization')
    plt.grid(True)
    plt.show()

# 选择一些有代表性的词进行可视化
words_to_visualize = [
    "中国", "北京", "上海",
    "自然语言处理", "人工智能", "深度学习",
    "词嵌入", "模型", "技术"
]

visualize_word_vectors(model, word_to_idx, words_to_visualize)

# 8. 词向量应用
def get_word_vector(model, word_to_idx, word):
    """获取词的向量表示"""
    if word not in word_to_idx:
        return None
    with torch.no_grad():
        word_idx = torch.LongTensor([word_to_idx[word]])
        return model.embeddings(word_idx).numpy()[0]

def find_similar_words(model, word_to_idx, idx_to_word, word, topn=5):
    """查找相似词"""
    if word not in word_to_idx:
        return []
    
    # 获取目标词的向量
    target_vector = get_word_vector(model, word_to_idx, word)
    if target_vector is None:
        return []
    
    # 计算与其他词的余弦相似度
    similarities = []
    for other_word, idx in word_to_idx.items():
        if other_word == word:
            continue
        other_vector = get_word_vector(model, word_to_idx, other_word)
        similarity = np.dot(target_vector, other_vector) / (
            np.linalg.norm(target_vector) * np.linalg.norm(other_vector))
        similarities.append((other_word, similarity))
    
    # 返回最相似的词
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:topn]

def word_analogy(model, word_to_idx, idx_to_word, positive, negative, topn=3):
    """词语类比"""
    # 计算目标向量
    target_vector = np.zeros_like(get_word_vector(model, word_to_idx, positive[0]))
    for word in positive:
        target_vector += get_word_vector(model, word_to_idx, word)
    for word in negative:
        target_vector -= get_word_vector(model, word_to_idx, word)
    
    # 计算相似度
    similarities = []
    for word, idx in word_to_idx.items():
        if word in positive or word in negative:
            continue
        vector = get_word_vector(model, word_to_idx, word)
        similarity = np.dot(target_vector, vector) / (
            np.linalg.norm(target_vector) * np.linalg.norm(vector))
        similarities.append((word, similarity))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:topn]

# 测试词语相似度
test_words = ["中国", "北京", "人工智能"]
for word in test_words:
    print(f"\n'{word}' 的相似词:")
    similar_words = find_similar_words(model, word_to_idx, idx_to_word, word)
    for similar_word, similarity in similar_words:
        print(f"{similar_word}: {similarity:.4f}")

# 测试词语类比
print("\n词语类比测试:")
result = word_analogy(model, word_to_idx, idx_to_word, 
                     positive=["北京", "中国"], 
                     negative=["上海"])
print("北京 : 中国 = 上海 : ?")
for word, similarity in result:
    print(f"{word}: {similarity:.4f}")

# 9. 总结
print("""
通过这个教程，我们学习了：
1. Word2Vec的基本原理
2. 如何使用PyTorch实现Word2Vec
3. 如何训练词向量模型
4. 如何可视化和应用词向量

这个实现虽然简单，但包含了Word2Vec的核心思想。在实际应用中，你可能需要：
1. 使用更大的语料库
2. 添加负采样
3. 使用更高级的优化器
4. 添加并行处理

希望这个教程对你理解Word2Vec有所帮助！
""") 