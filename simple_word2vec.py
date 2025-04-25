#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用Skip-gram模型
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import time
import jieba

class SimpleWord2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SimpleWord2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x):
        embeds = self.embeddings(x)
        out = self.linear(embeds)
        return out

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

#    print(word_to_idx)
#    print(idx_to_word)
    
    return word_to_idx, idx_to_word, len(word_counts)

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
    for epoch in range(epochs):
        total_loss = 0
        for center, context in data:
            # 准备输入和目标
            center_tensor = torch.LongTensor([center])
            context_tensor = torch.LongTensor([context])
            
            # 前向传播
            output = model(center_tensor)
#            print(output)  # 每个词作为上下文词的概率预测[词表长度, 1]
#            print(context_tensor) # 真实的索引 比如[9]
            loss = criterion(output, context_tensor) #交叉熵损失
            
            # 反向传播
            optimizer.zero_grad()# 上一步的梯度清零
            loss.backward()#反向传播计算梯度
            optimizer.step()#更新参数
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data):.4f}")
    
    end_time = time.time()
    print(f"训练完成，耗时: {end_time - start_time:.2f}秒")
    
    return model, word_to_idx, idx_to_word

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

if __name__ == "__main__":
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
    
    # 训练模型
    model, word_to_idx, idx_to_word = train_word2vec(
        sentences,
        embedding_dim=50,
        window_size=2,
        epochs=10,
        learning_rate=0.01
    )
    
    # 测试
    test_words = ["中国", "北京", "人工智能"]
    for word in test_words:
        print(f"\n'{word}' 的相似词:")
        similar_words = find_similar_words(model, word_to_idx, idx_to_word, word)
        for similar_word, similarity in similar_words:
            print(f"{similar_word}: {similarity:.4f}") 