"""
多目标优化评估指标模块
====================

该模块实现了多目标优化中常用的性能评估指标：
1. MED (Mean Euclidean Distance): 平均欧氏距离
2. IGD (Inverted Generational Distance): 反向世代距离

这些指标用于评估算法生成的Pareto前沿近似解与真实Pareto前沿的接近程度。
"""

import numpy as np
import torch


def MED(target_points, predict_points):
    """
    计算平均欧氏距离 (Mean Euclidean Distance)
    
    MED衡量预测点与目标点之间的平均欧氏距离，值越小表示预测越准确。
    
    Args:
        target_points (np.ndarray): 目标点集，形状为 (n_points, n_objectives)
        predict_points (np.ndarray): 预测点集，形状为 (n_points, n_objectives)
        
    Returns:
        float: 平均欧氏距离值
    """
    med = np.mean(np.sqrt(np.sum(np.square(target_points - predict_points), axis=1)))
    return med
    

def IGD(pf_truth, pf_approx):
    """
    计算反向世代距离 (Inverted Generational Distance)
    
    IGD衡量真实Pareto前沿上每个点到近似Pareto前沿的最小距离的平均值。
    值越小表示近似前沿越接近真实前沿，收敛性和分布性越好。
    
    Args:
        pf_truth (np.ndarray): 真实Pareto前沿，形状为 (n_true_points, n_objectives)
        pf_approx (np.ndarray): 近似Pareto前沿，形状为 (n_approx_points, n_objectives)
        
    Returns:
        float: IGD指标值
    """
    d_i = []
    for pf in pf_truth:
        # 计算真实前沿上每个点到近似前沿的最小距离
        d_i.append(np.min(np.sqrt(np.sum(np.square(pf - pf_approx), axis=1))))
    igd = np.mean(np.array(d_i))
    return igd