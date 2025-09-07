"""
预测和评估模块
==============

该模块实现超网络模型的预测和性能评估功能。

主要功能：
- 加载训练好的超网络模型
- 对给定偏好向量进行预测
- 计算评估指标（IGD, MED）
- 进行多次随机测试并统计结果

评估流程：
1. 加载对应约束条件下训练的模型
2. 随机生成偏好向量进行测试
3. 与真实Pareto前沿进行比较
4. 计算距离误差指标
"""

import torch
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import random
import argparse
from tools.utils import find_target, circle_points_random, get_d_paretomtl
from tools.utils import circle_points, sample_vec, vis_2d, vis_3d
from tools.metrics import IGD, MED
from matplotlib import pyplot as plt
import itertools
from matplotlib.tri import Triangulation, LinearTriInterpolator
from scipy import stats
import itertools
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib as mpl
import torch.nn.functional as F
import torch
from tqdm import tqdm
from itertools import product
from tools.utils import set_seed

class Arrow3D(FancyArrowPatch):
    """
    3D箭头类，用于在3D图中绘制箭头
    
    该类继承自matplotlib的FancyArrowPatch，扩展为支持3D显示。
    主要用于在3D可视化中绘制偏好向量和方向指示。
    """
    
    def __init__(self, xs, ys, zs, *args, **kwargs):
        """
        初始化3D箭头
        
        Args:
            xs: x坐标序列 [起点x, 终点x]
            ys: y坐标序列 [起点y, 终点y] 
            zs: z坐标序列 [起点z, 终点z]
        """
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """绘制箭头"""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
    def do_3d_projection(self, renderer=None):
        """执行3D投影"""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def count_parameters(model):
    """
    计算模型的可训练参数数量
    
    Args:
        model (torch.nn.Module): PyTorch模型
        
    Returns:
        int: 可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def get_rays(cfg, num_ray_init):
    """
    生成用于测试的偏好向量集合
    
    该函数生成满足特定条件的偏好向量，用于模型评估。
    过滤掉权重过小（<=0.16）的偏好向量，确保测试的有效性。
    
    Args:
        cfg (dict): 配置字典
        num_ray_init (int): 初始生成的偏好向量数量
        
    Returns:
        np.ndarray: 过滤后的偏好向量集合，形状为 (n_valid_rays, n_tasks)
    """
    # 生成初始偏好向量集合
    contexts = np.array(sample_vec(cfg['TRAIN']['N_task'], num_ray_init))
    tmp = []
    
    # 过滤偏好向量：移除权重过小的向量
    for r in contexts:
        flag = True
        for i in r:
            if i <= 0.16:  # 权重阈值
                flag = False
                break
        if flag:
            tmp.append(r)
    
    contexts = np.array(tmp)
    return contexts
def circle_points(K, min_angle=None, max_angle=None):
    """
    生成均匀分布的2D偏好向量（圆形分布）
    
    在指定角度范围内生成K个均匀分布的单位向量，用于2目标优化问题。
    
    Args:
        K (int): 生成的点数量
        min_angle (float, optional): 最小角度，默认为1e-6
        max_angle (float, optional): 最大角度，默认为π/2-1e-6
        
    Returns:
        np.ndarray: 形状为(K, 2)的偏好向量数组
    """
    ang0 = 1e-6 if min_angle is None else min_angle
    ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]


def simplex(n_vals):
    """
    生成3D单纯形上的均匀分布点
    
    用于3目标优化问题的偏好向量生成，生成满足单纯形约束的点。
    
    Args:
        n_vals (int): 每个维度的采样点数量
        
    Returns:
        np.ndarray: 单纯形上的坐标点数组
    """
    base = np.linspace(0, 0.25, n_vals, endpoint=False)
    coords = np.asarray(list(itertools.product(base, repeat=3)))
    return coords[np.isclose(coords.sum(axis=-1), 0.25)]
def predict_result(device, cfg, criterion, pb, pf, model_type):
    """
    执行模型预测和性能评估
    
    该函数加载训练好的模型，对随机生成的偏好向量进行预测，
    并计算与真实Pareto前沿的距离误差。通过多次随机测试评估模型性能。
    
    Args:
        device (torch.device): 计算设备
        cfg (dict): 配置字典
        criterion (str): 标量化函数类型
        pb (Problem): 问题实例
        pf (np.ndarray): 真实Pareto前沿
        model_type (str): 模型类型 ('mlp' 或 'trans')
    """
    
    print(f"模型类型: {model_type}")
    
    # 从配置中提取评估参数
    mode = cfg['MODE']                              # 维度模式
    name = cfg['NAME']                              # 问题名称
    print(f"问题: {name}")
    
    num_ray_init = cfg['EVAL']['Num_ray_init']      # 初始偏好向量数量
    num_ray_test = cfg['EVAL']['Num_ray_test']      # 测试偏好向量数量
    out_dim = cfg['TRAIN']['Out_dim']               # 输出维度
    ray_hidden_dim = cfg['TRAIN']['Ray_hidden_dim'] # 隐藏层维度
    n_tasks = cfg['TRAIN']['N_task']                # 目标函数数量
    # 设置与训练时相同的约束条件
    # 每个约束条件对应一个训练好的模型
    if name == 'ex1':
        # CVX1问题的约束条件设置
        c_s = [[0, 0.8], [0.1, 0.6], [0.2, 0.4], [0.35, 0.22], [0.6, 0.1]]
    elif name == 'ex2':
        # CVX2问题的约束条件设置
        c_s = [[0, 0.6], [0.02, 0.4], [0.16, 0.2], [0.2, 0.15], [0.4, 0.02]]
    elif name == 'ex3':
        # CVX3问题（3目标）的约束条件设置
        c_s = [[0.15, 0.2, 0.7], [0.2, 0.5, 0.6], [0.2, 0.7, 0.4], [0.35, 0.6, 0.22], [0.6, 0.1, 0.46]]
    elif name == 'ZDT1':
        # ZDT1测试函数的约束条件设置
        c_s = [[0, 0.8], [0.1, 0.6], [0.2, 0.4], [0.35, 0.22], [0.6, 0.1]]
    elif name == 'ZDT2':
        # ZDT2测试函数的约束条件设置
        c_s = [[0.1, 0.9], [0.1, 0.6], [0.2, 0.4], [0.35, 0.22], [0.6, 0.1]]
    elif name == 'DTLZ2':
        # DTLZ2测试函数（3目标）的约束条件设置
        c_s = [[0.15, 0.2, 0.7], [0.2, 0.5, 0.6], [0.2, 0.7, 0.4], [0.35, 0.6, 0.22], [0.6, 0.1, 0.46]]
    count = 0
    for se in tqdm(range(30)):
        count += 1
        set_seed(se)
        meds_se = []
        for c_ in c_s:
            hnet = torch.load("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_" + str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+".pt",map_location=device,weights_only=False)
            meds_c = []
            if count == 1:
                print(count_parameters(hnet))
            hnet.eval()
            hnet = hnet.to(device)
            results1 = []
            targets_epo = []
            for i in range(10):  
                c_in = torch.Tensor(c_).to(device)
                if n_tasks == 2:
                    u1 = random.uniform(c_[0], 1)
                    u2 = random.uniform(c_[1], 1)
                    u = np.array([u1,u2])

                else:
                    u1 = random.uniform(c_[0], 1)
                    u2 = random.uniform(c_[1], 1)
                    u3 = random.uniform(c_[2], 1)
                    u = np.array([u1,u2,u3])
                r = u/np.linalg.norm(u,1)
                #tmp.append(r)
                ray = torch.from_numpy(r).float()
                output = hnet(ray)
                if model_type == 'trans':
                    if cfg["NAME"] == "ex3":
                        output = F.softmax(output,dim=1)
                    else:
                        output = F.sigmoid(output)
                    #output = torch.mean(output,dim=0)  
                    #output = output.unsqueeze(0) 
                    #print(output)
                else:
                    output = output.unsqueeze(0)
                    if cfg["NAME"] == "ex3":
                        
                        output = F.softmax(output,dim=1)
                    else:
                        output = F.sigmoid(output)
                    
                if cfg["NAME"] == "ex2":
                    output = 5*output
                elif cfg["NAME"] == "ex3":
                    
                    output = torch.sqrt(output)


                objectives = pb.get_values(output)
                obj_values = []
                
                for j in range(len(objectives)):
                    obj_values.append(objectives[j].cpu().detach().numpy().tolist())
                results1.append(obj_values)
                target_epo = find_target(pf, criterion = criterion, context = r.tolist(),c=c_,cfg=cfg)
                targets_epo.append(target_epo)
            targets_epo = np.array(targets_epo)
            results1 = np.array(results1, dtype='float32')
            med = np.mean(np.sqrt(np.sum(np.square(targets_epo-results1),axis = 1)))
            med = MED(targets_epo, results1)
            #print(med)
            meds_c.append(med)

            d_i = []
            for target in pf:
                d_i.append(np.min(np.sqrt(np.sum(np.square(target-results1),axis = 1))))
            igd = np.mean(np.array(d_i))
            
            igd = IGD(pf, results1)

            #contexts = np.array(tmp)
            
            # 创建predict目录（如果不存在）
            os.makedirs('./predict', exist_ok=True)
            
            # 保存结果数据
            np.save('./predict/target_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'.npy',targets_epo)
            np.save('./predict/predict_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'.npy',results1)
            np.save('./predict/med_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'.npy',med)
            
            # 重新生成偏好向量用于保存
            tmp_save = []
            for i in range(10):
                if n_tasks == 2:
                    u1 = random.uniform(c_[0], 1)
                    u2 = random.uniform(c_[1], 1)
                    u = np.array([u1,u2])
                else:
                    u1 = random.uniform(c_[0], 1)
                    u2 = random.uniform(c_[1], 1)
                    u3 = random.uniform(c_[2], 1)
                    u = np.array([u1,u2,u3])
                r = u/np.linalg.norm(u,1)
                tmp_save.append(r)
            contexts_save = np.array(tmp_save)
            np.save('./predict/ray_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'.npy',contexts_save)
            meds_se.append(np.mean(np.array(meds_c).tolist()))
    print("Mean: ",np.mean(np.array(meds_se)))
    print("Std: ",np.std(np.array(meds_se)))
    
    # 添加可视化功能
    # 使用最后一组结果进行可视化展示  
    show_visualization = cfg.get('SHOW_VIZ', False)
    if show_visualization and len(targets_epo) > 0 and len(results1) > 0:
        print("\n正在生成可视化图表...")
        
        # 重新生成偏好向量用于可视化
        tmp = []
        for i in range(10):
            if n_tasks == 2:
                u1 = random.uniform(c_[0], 1)
                u2 = random.uniform(c_[1], 1)
                u = np.array([u1,u2])
            else:
                u1 = random.uniform(c_[0], 1)
                u2 = random.uniform(c_[1], 1)
                u3 = random.uniform(c_[2], 1)
                u = np.array([u1,u2,u3])
            r = u/np.linalg.norm(u,1)
            tmp.append(r)
        contexts = np.array(tmp)
        
        # 根据目标数量选择合适的可视化函数
        if n_tasks == 2:
            # 2D可视化
            vis_2d(cfg, targets_epo, results1, contexts, pb, pf, criterion, 
                   igd, med, c_, model_type, False)
        elif n_tasks == 3:
            # 3D可视化  
            vis_3d(cfg, targets_epo, results1, contexts, pb, pf, criterion,
                   igd, med, model_type, False)
