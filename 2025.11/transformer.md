1.the basic architecture of transformer

![transformer](../image/transformer.png)

2.encoder与decoder架构，主要的就是注意力机制，论文链接https://arxiv.org/abs/1706.03762，进行了机器翻译的实验来验证模型效果BLEU score，英语翻译德语、法语，后续用于nlp，vit，可以并行(需要理解为什么，如何做到的)

回顾李沐讲解视频，卷积神经网络对长序列的效果需要多个卷积进行计算才能将两个较远的点融合靠近，提出了自注意力机制（第一次使用）以及MHA

3.endoer：将输入变为一组连续的向量表示，n为6，即6个encoder复合计算，每一个bolck包含包含mha与ffn网络两个子层，中间穿插resadd与layernorm

layernorm 与batchnorm的区别，layernorm多一点，当样本长度变化大时，均值与方差波动会比较大，layernorm由于只关注自身的，不会太受影响

attetion机制：Q K V 的加权和
$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} + M \right) V
$$
4.decoder：根据编码结果逐步生成输出,(根据任务),一个个按顺序生成，自回归的方式










