"""
超网络模型定义
============

该文件定义了两种超网络架构用于多目标优化：
1. Hypernet_mlp: 基于多层感知机的超网络
2. Hypernet_trans: 基于Transformer的超网络

超网络(HyperNetwork)的作用：
- 接收偏好向量(ray/preference vector)作为输入
- 输出对应该偏好的决策变量
- 通过训练学习Pareto前沿上不同点的映射关系
"""

from torch import nn
import torch.nn.functional as F
import torch

class Hypernet_mlp(nn.Module):
    """
    基于多层感知机的超网络
    
    该模型使用深层MLP来学习从偏好向量到决策变量的映射关系。
    网络结构：7层全连接层，每层使用ReLU激活函数。
    
    Args:
        ray_hidden_dim (int): 隐藏层维度，默认30
        out_dim (int): 输出维度（决策变量维度），默认1  
        target_hidden_dim (int): 目标隐藏层维度，默认15（当前未使用）
        n_hidden (int): 隐藏层数量，默认1（当前未使用）
        n_tasks (int): 目标函数数量，默认2
    """
    
    def __init__(self, ray_hidden_dim=30, out_dim=1, target_hidden_dim=15, n_hidden=1, n_tasks=2):
        super().__init__()
        self.n_hidden = n_hidden  # 隐藏层数量
        self.n_tasks = n_tasks    # 目标函数数量
        self.out_dim = out_dim    # 输出维度
        
        # 7层深度MLP网络，用于学习偏好向量到决策变量的映射
        self.ray_mlp = nn.Sequential(
                nn.Linear(self.n_tasks, ray_hidden_dim),    # 输入层：偏好向量 -> 隐藏层
                nn.ReLU(inplace=True),
                nn.Linear(ray_hidden_dim, ray_hidden_dim),  # 隐藏层1
                nn.ReLU(inplace=True),
                nn.Linear(ray_hidden_dim, ray_hidden_dim),  # 隐藏层2
                nn.ReLU(inplace=True),
                nn.Linear(ray_hidden_dim, ray_hidden_dim),  # 隐藏层3
                nn.ReLU(inplace=True),
                nn.Linear(ray_hidden_dim, ray_hidden_dim),  # 隐藏层4
                nn.ReLU(inplace=True),
                nn.Linear(ray_hidden_dim, ray_hidden_dim),  # 隐藏层5
                nn.ReLU(inplace=True),
                nn.Linear(ray_hidden_dim, ray_hidden_dim),  # 隐藏层6
                nn.ReLU(inplace=True),
                nn.Linear(ray_hidden_dim, out_dim),         # 输出层：决策变量
            )

    def forward(self, ray):
        """
        前向传播
        
        Args:
            ray (torch.Tensor): 偏好向量，形状为 (n_tasks,)
            
        Returns:
            torch.Tensor: 决策变量，形状为 (out_dim,)
        """
        x = self.ray_mlp(ray)
        return x

class Hypernet_trans(nn.Module):
    """
    基于Transformer的超网络
    
    该模型使用Transformer架构来学习偏好向量各分量之间的关系。
    主要特点：
    - 为每个目标函数分量设计独立的嵌入层
    - 使用多头注意力机制捕获偏好向量内部关系  
    - 采用前馈网络和残差连接提升表达能力
    
    Args:
        ray_hidden_dim (int): 隐藏层维度，默认30
        out_dim (int): 输出维度（决策变量维度），默认1
        target_hidden_dim (int): 目标隐藏层维度，默认15（当前未使用）
        n_hidden (int): 隐藏层数量，默认1（当前未使用）
        n_tasks (int): 目标函数数量，默认2
    """
    
    def __init__(self, ray_hidden_dim=30, out_dim=1, target_hidden_dim=15, n_hidden=1, n_tasks=2):
        super().__init__()
        self.n_hidden = n_hidden  # 隐藏层数量
        self.n_tasks = n_tasks    # 目标函数数量

        # 根据目标函数数量创建对应的嵌入层
        if self.n_tasks == 2:
            # 2目标优化：为每个目标创建独立的嵌入层
            self.embedding_layer1 = nn.Sequential(nn.Linear(1, ray_hidden_dim), nn.ReLU(inplace=True))
            self.embedding_layer2 = nn.Sequential(nn.Linear(1, ray_hidden_dim), nn.ReLU(inplace=True))
        else:
            # 3目标优化：为每个目标创建独立的嵌入层
            self.embedding_layer1 = nn.Sequential(nn.Linear(1, ray_hidden_dim), nn.ReLU(inplace=True))
            self.embedding_layer2 = nn.Sequential(nn.Linear(1, ray_hidden_dim), nn.ReLU(inplace=True))
            self.embedding_layer3 = nn.Sequential(nn.Linear(1, ray_hidden_dim), nn.ReLU(inplace=True))
            
        # 输出层：将注意力机制处理后的特征映射到决策变量
        self.output_layer = nn.Linear(ray_hidden_dim, out_dim)
        
        # 多头注意力机制：学习偏好向量各分量间的关系
        self.attention = nn.MultiheadAttention(embed_dim=ray_hidden_dim, num_heads=2)
        
        # 前馈网络：进一步处理注意力输出
        self.ffn1 = nn.Linear(ray_hidden_dim, ray_hidden_dim)
        self.ffn2 = nn.Linear(ray_hidden_dim, ray_hidden_dim)


    def forward(self, ray):
        """
        前向传播
        
        处理流程：
        1. 将偏好向量的每个分量通过独立的嵌入层
        2. 使用多头注意力机制学习分量间关系
        3. 通过前馈网络进一步处理
        4. 应用残差连接增强梯度流
        5. 平均池化并输出最终决策变量
        
        Args:
            ray (torch.Tensor): 偏好向量，形状为 (n_tasks,)
            
        Returns:
            torch.Tensor: 决策变量，形状为 (out_dim,)
        """
        ray = ray.unsqueeze(0)  # 增加batch维度: (1, n_tasks)
        
        # 将偏好向量的每个分量通过对应的嵌入层
        if self.n_tasks == 2: 
            # 2目标：分别嵌入两个偏好分量
            x = torch.stack((
                self.embedding_layer1(ray[:, 0].unsqueeze(1)),
                self.embedding_layer2(ray[:, 1].unsqueeze(1))
            ))
        else:
            # 3目标：分别嵌入三个偏好分量
            x = torch.stack((
                self.embedding_layer1(ray[:, 0].unsqueeze(1)),
                self.embedding_layer2(ray[:, 1].unsqueeze(1)),
                self.embedding_layer3(ray[:, 2].unsqueeze(1))
            ))
        
        x_ = x  # 保存用于残差连接
                
        # 多头注意力机制：学习偏好分量间的交互关系
        x, _ = self.attention(x, x, x)
        x = x + x_  # 残差连接1
        
        x_ = x  # 保存用于下一个残差连接
        
        # 前馈网络处理
        x = self.ffn1(x)
        x = F.relu(x)
        x = self.ffn2(x)
        x = x + x_  # 残差连接2
        
        # 输出层和平均池化
        x = self.output_layer(x)
        x = torch.mean(x, dim=0)  # 在序列维度上平均池化
        
        return x