"""
主程序入口文件
==============

该文件是CPFL_MOP (Connected Pareto Front Multi-Objective Optimization) 项目的主入口，
用于训练和测试超网络模型解决多目标优化问题。

项目功能：
- 支持多种多目标优化问题（ex1-ex4, ZDT1/2, DTLZ2）
- 支持多种标量化函数（LS, KL, Cheby, Utility, Cosine等）
- 支持MLP和Transformer两种超网络架构
- 提供训练和预测两种运行模式

使用方法：
    python main.py --solver LS --problem ex1 --mode train --model_type mlp
"""

import sys
import os
sys.path.append(os.getcwd())
import time
import numpy as np
import torch
from tools.utils import visualize_2d, visualize_3d, visualize_predict_2d, visualize_predict_3d,concat_2d,vis_2d,vis_3d
import argparse
import yaml
from problems.get_problem import Problem
from train import train_epoch
from predict import predict_result
from itertools import product

def run_train(cfg, criterion, device, problem, model_type):
    """
    运行训练过程
    
    Args:
        cfg (dict): 配置字典，包含训练参数
        criterion (str): 标量化函数类型 (LS, KL, Cheby等)
        device (torch.device): 计算设备 (CPU或GPU)
        problem (str): 问题名称 (ex1, ex2, ZDT1等)
        model_type (str): 模型类型 (mlp或trans)
    
    Returns:
        训练过程中生成的解的列表
    """
    pb = Problem(problem, cfg['MODE'])  # 创建问题实例
    pf = pb.get_pf()  # 获取真实Pareto前沿
    if cfg['MODE'] == '2d':
        sol = train_epoch(device, cfg, criterion, pb, pf, model_type)
    else:
        sol = train_epoch(device, cfg, criterion, pb, pf, model_type)


def run_predict(cfg, criterion, device, problem, model_type, show_viz=True):
    """
    运行预测/测试过程
    
    Args:
        cfg (dict): 配置字典，包含测试参数
        criterion (str): 标量化函数类型
        device (torch.device): 计算设备
        problem (str): 问题名称
        model_type (str): 模型类型
        show_viz (bool): 是否显示可视化
    """
    pb = Problem(problem, cfg['MODE'])  # 创建问题实例
    pf = pb.get_pf()  # 获取真实Pareto前沿
    
    # 设置可视化标志
    cfg['SHOW_VIZ'] = show_viz
    
    if cfg['MODE'] == '2d':   
        predict_result(device, cfg, criterion, pb, pf, model_type)
    else:    
        predict_result(device, cfg, criterion, pb, pf, model_type)

if __name__ == "__main__":
    # 设置计算设备（优先使用GPU）
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
    
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="CPFL_MOP 多目标优化项目")
    
    # 标量化函数选择
    parser.add_argument(
        "--solver", type=str, 
        choices=["LS", "KL", "Cheby", "Utility", "Cosine", "Cauchy", "Prod", "Log", "AC", "MC", "HV", "CPMTL", "EPO", "HVI"],
        default="Cheby", 
        help="选择标量化函数: LS(线性), KL(KL散度), Cheby(切比雪夫), Utility(效用函数)等"
    )
    
    # 优化问题选择
    parser.add_argument(
        "--problem", type=str, 
        choices=["ex1", "ex2", "ex3", "ex4", "ZDT1", "ZDT2", "DTLZ2"],
        default="ex1", 
        help="选择优化问题: ex1-ex4(示例问题), ZDT1/ZDT2(ZDT测试函数), DTLZ2(DTLZ测试函数)"
    )
    
    # 运行模式
    parser.add_argument(
        "--mode", type=str, 
        default="test",
        help="运行模式: train(训练模式) 或 test(测试模式)"
    )
    
    # 模型类型
    parser.add_argument(
        "--model_type", type=str, 
        default="mlp",
        help="模型类型: mlp(多层感知机) 或 trans(Transformer)"
    )
    
    # 可视化控制
    parser.add_argument(
        "--visualize", action="store_true",
        help="是否显示可视化图表"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    criterion = args.solver 
    model_type = args.model_type
    problem = args.problem
    
    # 打印运行配置信息
    print(f"模型类型: {model_type}")
    print(f"标量化函数: {criterion}")
    print(f"优化问题: {problem}")
    print(f"运行模式: {args.mode}")
    print(f"计算设备: {device}")
    
    # 加载配置文件
    config_file = f"./configs/{problem}.yaml"
    with open(config_file) as stream:
        cfg = yaml.safe_load(stream)
    
    # 根据运行模式执行相应功能
    if args.mode == "train":
        print("开始训练...")
        run_train(cfg, criterion, device, problem, model_type)
    else:
        print("开始预测...")
        run_predict(cfg, criterion, device, problem, model_type, args.visualize)