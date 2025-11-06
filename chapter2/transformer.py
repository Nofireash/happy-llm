import torch
import math

"""计算注意力函数"""
def attention(query, key, value, dropout=None):
    '''
    args:
    query: 查询值矩阵
    key: 键值矩阵
    value: 值矩阵
    '''
    # 获取键向量的维度，键向量的维度和值向量的维度相同
    d_k = query.size(-1)
    # 计算Q与K的内积并除以根号dk
    # transpose——相当于转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) 
    # 测试提交代码


import torch
import math
from torch import nn
from dataclasses import dataclass
from transformers import BertTokenizer
import torch.nn.functional as F

@dataclass
class ModelArgs:
    n_embd: int # 嵌入维度
    n_heads: int # 头数
    

'''多头自注意力计算模块'''
class MultiHeadAttention(nn.modules):

    def init(self, args: ModelArgs, is_causal=False):
        # 构造函数
        # args: 配置对象
        super().__init__()
