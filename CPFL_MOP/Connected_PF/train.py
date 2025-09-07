"""
训练模块
========

该模块实现超网络的训练过程，用于学习多目标优化问题的Pareto前沿。

主要功能：
- 使用切比雪夫标量化函数训练超网络
- 支持多种约束条件设置（不同的下界）
- 实现MLP和Transformer两种超网络架构的训练
- 自动保存训练好的模型权重

训练策略：
- 对每个约束条件单独训练一个模型
- 使用随机采样的偏好向量进行训练
- 应用不同的激活函数处理不同问题类型
"""

import sys
import os
sys.path.append(os.getcwd())
import time
from tqdm import tqdm
import numpy as np
import torch
from tools.scalarization_function import CS_functions, EPOSolver
from tools.hv import HvMaximization
from models import Hypernet_mlp, Hypernet_trans
from tools.utils import set_seed
import random
import torch.nn.functional as F
import torch
from predict import predict_result
from itertools import product

def sample_config(search_space_dict, reset_random_seed=False, seed=0):
    """
    从搜索空间中随机采样配置参数
    
    Args:
        search_space_dict (dict): 搜索空间字典
        reset_random_seed (bool): 是否重置随机种子
        seed (int): 随机种子值
        
    Returns:
        dict: 随机采样的配置
    """
    if reset_random_seed:
        random.seed(seed)
    
    config = dict()
    
    for key, value in search_space_dict.items():
        config[key] = random.choice(value)
        
    return config


def count_parameters(model):
    """
    计算模型的可训练参数数量
    
    Args:
        model (torch.nn.Module): PyTorch模型
        
    Returns:
        int: 可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def train_epoch(device, cfg, criterion, pb, pf, model_type):
    """
    执行一个完整的训练过程
    
    该函数实现多目标优化超网络的训练，针对不同的约束条件分别训练模型。
    训练策略：对每个约束条件（下界）单独训练一个超网络模型。
    
    Args:
        device (torch.device): 计算设备
        cfg (dict): 配置字典，包含所有训练参数
        criterion (str): 标量化函数类型
        pb (Problem): 问题实例，用于计算目标函数值
        pf (np.ndarray): 真实Pareto前沿
        model_type (str): 模型类型 ('mlp' 或 'trans')
        
    Returns:
        list: 训练过程中生成的解的列表
    """
    # 设置随机种子确保结果可复现
    set_seed(42)
    
    # 从配置中提取训练参数
    name = cfg['NAME']                           # 问题名称
    mode = cfg['MODE']                          # 维度模式 ('2d' 或 '3d')
    ray_hidden_dim = cfg['TRAIN']['Ray_hidden_dim']     # 隐藏层维度
    out_dim = cfg['TRAIN']['Out_dim']                   # 输出维度
    n_tasks = cfg['TRAIN']['N_task']                    # 目标函数数量
    num_hidden_layer = cfg['TRAIN']['Solver'][criterion]['Num_hidden_layer']  # 隐藏层数量
    last_activation = cfg['TRAIN']['Solver'][criterion]['Last_activation']    # 最后激活函数
    ref_point = tuple(map(int, cfg['TRAIN']['Ref_point'].split(',')))         # 参考点
    
    # 优化器参数
    lr = cfg['TRAIN']['OPTIMIZER']['Lr']                # 学习率
    wd = cfg['TRAIN']['OPTIMIZER']['WEIGHT_DECAY']      # 权重衰减
    type_opt = cfg['TRAIN']['OPTIMIZER']['TYPE']        # 优化器类型
    epochs = cfg['TRAIN']['Epoch']                      # 训练轮数
    alpha_r = cfg['TRAIN']['Alpha']                     # Alpha参数
    start = 0.
    # 为不同问题设置约束条件（下界设置）
    # 每个约束条件代表Pareto前沿的一个限制区域
    if name == 'ex1':
        # CVX1问题：5个不同的下界设置
        c_s = [[0, 0.8], [0.1, 0.6], [0.2, 0.4], [0.35, 0.22], [0.6, 0.1]]
    elif name == 'ex2':
        # CVX2问题：5个不同的下界设置  
        c_s = [[0, 0.6], [0.02, 0.4], [0.16, 0.2], [0.2, 0.15], [0.4, 0.02]]
    elif name == 'ex3':
        # CVX3问题（3目标）：5个不同的下界设置
        c_s = [[0.15, 0.2, 0.7], [0.2, 0.5, 0.6], [0.2, 0.7, 0.4], [0.35, 0.6, 0.22], [0.6, 0.1, 0.46]]
    elif name == 'ZDT1':
        # ZDT1测试函数：5个不同的下界设置
        c_s = [[0, 0.8], [0.1, 0.6], [0.2, 0.4], [0.35, 0.22], [0.6, 0.1]]
    elif name == 'ZDT2':
        # ZDT2测试函数：5个不同的下界设置
        c_s = [[0.1, 0.9], [0.1, 0.6], [0.2, 0.4], [0.35, 0.22], [0.6, 0.1]]
    elif name == 'DTLZ2':
        # DTLZ2测试函数（3目标）：5个不同的下界设置
        c_s = [[0.15, 0.2, 0.7], [0.2, 0.5, 0.6], [0.2, 0.7, 0.4], [0.35, 0.6, 0.22], [0.6, 0.1, 0.46]]
    
    
    sol = []  # 存储所有训练解
    
    # 对每个约束条件分别训练一个超网络模型
    for c_ in c_s:
        print(f"训练约束条件: {c_}")
        
        # 根据模型类型创建超网络
        if model_type == "mlp":
            hnet = Hypernet_mlp(
                ray_hidden_dim=ray_hidden_dim, 
                out_dim=out_dim, 
                target_hidden_dim=ray_hidden_dim, 
                n_hidden=1, 
                n_tasks=n_tasks
            )
        else:
            hnet = Hypernet_trans(
                ray_hidden_dim=ray_hidden_dim, 
                out_dim=out_dim, 
                target_hidden_dim=ray_hidden_dim, 
                n_hidden=1, 
                n_tasks=n_tasks
            )
        
        # 将模型移到指定设备
        hnet = hnet.to(device)
        param = count_parameters(hnet)
        print(f"模型参数数量: {param}")
        
        # 设置优化器
        if type_opt == 'adam':
            optimizer = torch.optim.Adam(hnet.parameters(), lr=lr, weight_decay=wd) 
        elif type_opt == 'adamw':
            optimizer = torch.optim.AdamW(hnet.parameters(), lr=lr, weight_decay=wd)
        # 开始训练循环
        for epoch in tqdm(range(epochs), desc=f"训练 {c_}"):
            
            # 将约束条件转换为张量
            c = torch.tensor(c_)
            
            # 随机生成偏好向量（在约束条件范围内）
            if n_tasks == 2:
                # 2目标优化：生成2维偏好向量
                u1 = random.uniform(c[0], 1)  # 第一个目标的权重
                u2 = random.uniform(c[1], 1)  # 第二个目标的权重
                u = np.array([u1, u2])
            else:
                # 3目标优化：生成3维偏好向量
                u1 = random.uniform(c[0], 1)  # 第一个目标的权重
                u2 = random.uniform(c[1], 1)  # 第二个目标的权重
                u3 = random.uniform(c[2], 1)  # 第三个目标的权重
                u = np.array([u1, u2, u3])
            
            # 将偏好向量归一化（L1范数）
            lda = (u / np.linalg.norm(u, 1))
            ray = torch.from_numpy(lda).float()
            
            # 设置模型为训练模式
            hnet.train()
            optimizer.zero_grad()

            # 前向传播：通过超网络生成决策变量
            output = hnet(ray)
            
            # 根据模型类型和问题类型应用不同的激活函数
            if model_type == 'trans':
                if cfg["NAME"] == "ex3":
                    # ex3问题使用softmax激活（确保输出和为1）
                    output = F.softmax(output, dim=1)
                else:
                    # 其他问题使用sigmoid激活（输出范围[0,1]）
                    output = F.sigmoid(output)   
            else:
                # MLP模型需要增加batch维度
                output = output.unsqueeze(0)
                if cfg["NAME"] == "ex3":
                    # ex3问题使用softmax激活
                    output = F.softmax(output, dim=1)
                else:
                    # 其他问题使用sigmoid激活
                    output = F.sigmoid(output)
            
            # 根据问题类型对输出进行后处理
            if cfg["NAME"] == "ex2":
                # ex2问题：将输出缩放5倍
                output = 5 * output
            elif cfg["NAME"] == "ex3":
                # ex3问题：对输出开平方根
                output = torch.sqrt(output)

            # 计算目标函数值
            objectives = pb.get_values(output)
            obj_values = []
            for i in range(len(objectives)):
                obj_values.append(objectives[i])
            losses = torch.stack(obj_values)
            
            # 使用切比雪夫标量化函数计算损失
            CS_func = CS_functions(losses, ray)
            loss = CS_func.chebyshev_function(c)
            
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()
            
            # 保存当前训练步的目标函数值
            tmp = []
            for i in range(len(objectives)):
                tmp.append(objectives[i].cpu().detach().numpy().tolist())
            sol.append(tmp)

        # 保存训练好的模型
        model_path = f"./save_weights/best_weight_{criterion}_{mode}_{name}_{model_type}_{c_[0]}_{c_[1]}.pt"
        torch.save(hnet, model_path)
        print(f"模型已保存至: {model_path}")
    
    return sol  # 返回所有训练过程中的解


