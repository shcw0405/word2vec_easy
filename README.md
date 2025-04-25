# CPU 上运行 Word2Vec 示例

这个项目演示了如何在 CPU 上训练和使用 Word2Vec 模型进行词嵌入。Word2Vec 是一种经典的词嵌入技术，可以有效地将词语映射到向量空间，捕捉词语之间的语义关系。

## 环境要求

- Python 3.6+
- Gensim 4.0+
- NumPy

## 安装依赖

```bash
pip install gensim numpy
```

## 项目文件

- `word2vec_cpu_example.py` - 主程序，包含训练和测试 Word2Vec 模型的完整代码
- `sample_text.txt` - 自动生成的示例文本数据（第一次运行程序时会创建）

## 使用方法

1. 直接运行 Python 脚本:

```bash
python word2vec_cpu_example.py
```

这将:

- 生成示例文本文件（如果不存在）
- 训练 Word2Vec 模型
- 测试词向量效果
- 展示如何应用训练好的模型

## 自定义训练

如果要使用自己的数据训练 Word2Vec 模型，您需要:

1. 准备文本语料库，每行一个句子，词语之间用空格分隔
2. 修改脚本中的参数:
   - `input_file` - 输入文本文件路径
   - `output_model` - 模型保存路径
   - `output_vectors` - 词向量保存路径
   - `size` (或`vector_size`) - 词向量维度
   - `window` - 上下文窗口大小
   - `min_count` - 最小词频阈值
   - `workers` - CPU 工作线程数量

## CPU 优化提示

在 CPU 上运行 Word2Vec 时，可以通过以下方式提高性能:

1. 适当降低向量维度（如 50-100 维）
2. 减小训练语料的规模（如必要时随机抽样）
3. 利用多核 CPU 进行并行计算，设置合理的`workers`参数
4. 对于大型语料库，考虑使用增量训练方式

## 参考资料

- [Gensim Word2Vec 文档](https://radimrehurek.com/gensim/models/word2vec.html)
- [Word2Vec 论文](https://arxiv.org/abs/1301.3781)
