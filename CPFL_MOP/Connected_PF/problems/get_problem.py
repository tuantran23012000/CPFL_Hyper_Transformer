"""
多目标优化问题定义模块
======================

该模块定义了各种多目标优化测试问题，包括：
1. 经典测试函数：ZDT1, ZDT2, DTLZ2
2. 示例问题：ex1, ex2, ex3, ex4
3. 测试问题：test

每个问题类都包含：
- create_pf(): 创建真实Pareto前沿
- f_1(), f_2(), f_3(): 目标函数定义
- Problem类：统一的问题接口

主要功能：
- 提供标准化的多目标优化问题接口
- 生成真实Pareto前沿用于性能评估
- 计算给定决策变量的目标函数值
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymoo.util.remote import Remote


def get_ref_dirs(n_obj):
    """
    获取参考方向向量
    
    用于生成多目标优化问题的参考方向，支持2目标和3目标优化。
    
    Args:
        n_obj (int): 目标函数数量 (2 或 3)
        
    Returns:
        np.ndarray: 参考方向向量数组
        
    Raises:
        Exception: 当目标数量超过3时抛出异常
    """
    if n_obj == 2:
        # 2目标：生成100个均匀分布的参考点
        ref_dirs = UniformReferenceDirectionFactory(2, n_points=100).do()
    elif n_obj == 3:
        # 3目标：使用100个分区生成参考方向
        ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=100).do()
    else:
        raise Exception("目前仅支持2目标和3目标优化问题！")
    return ref_dirs


def generic_sphere(ref_dirs):
    """
    将参考方向投影到单位球面上
    
    Args:
        ref_dirs (np.ndarray): 参考方向向量
        
    Returns:
        np.ndarray: 单位球面上的方向向量
    """
    return ref_dirs / np.tile(np.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))
class DTLZ2():
    """
    DTLZ2测试函数（3目标优化问题）
    
    DTLZ2是经典的多目标优化测试函数，具有以下特点：
    - 3个目标函数
    - Pareto前沿为单位球面的一部分
    - 具有可扩展的决策变量维度
    """
    
    def __init__(self):
        """初始化DTLZ2问题"""
        self.a = 1  # 参数a，用于控制目标函数形状
    
    def create_pf(self):
        """
        创建DTLZ2的真实Pareto前沿
        
        Returns:
            np.ndarray: 真实Pareto前沿点集，形状为(n_points, 3)
        """
        ref_dirs = get_ref_dirs(3)  # 获取3目标的参考方向
        pf = generic_sphere(ref_dirs)  # 投影到单位球面
        return pf
    
    def f_1(self, output):
        """
        第一个目标函数
        
        Args:
            output (torch.Tensor): 决策变量，形状为(1, n_vars)
            
        Returns:
            torch.Tensor: 第一个目标函数值
        """
        return (torch.cos(torch.pi/2*output[0, 0]) * 
                torch.cos(torch.pi/2*output[0, 1]) * 
                (sum((output[0, 2:]-0.5)**2)+1))
    
    def f_2(self, output):
        """
        第二个目标函数
        
        Args:
            output (torch.Tensor): 决策变量，形状为(1, n_vars)
            
        Returns:
            torch.Tensor: 第二个目标函数值
        """
        return (torch.cos(torch.pi/2*output[0, 0]**self.a) * 
                torch.sin(torch.pi/2*output[0, 1]**self.a) * 
                (sum((output[0, 2:]-0.5)**2)+1))
    
    def f_3(self, output):
        """
        第三个目标函数
        
        Args:
            output (torch.Tensor): 决策变量，形状为(1, n_vars)
            
        Returns:
            torch.Tensor: 第三个目标函数值
        """
        return (torch.sin(torch.pi/2*output[0, 0]**self.a) * 
                (sum((output[0, 2:]-0.5)**2)+1))
class test():
    def __init__(self):
        #self.a = 1
        self.num = 1000
    def create_pf(self):
        ps = np.linspace(-10,10,num = self.num)
        pf = []
        for x1 in ps:
            for x2 in ps:
                x = torch.Tensor([[x1,x2]])
                f= np.stack([self.f_1(x).item(),self.f_2(x).item()])
                pf.append(f)   
        pf = np.array(pf)
        return pf
    def f_1(self, output):
        return torch.cos(output[0, 0])**2 + 0.2
    def f_2(self, output):
        return 1.3+torch.sin(output[0, 1])**2- torch.cos(output[0, 0])-0.1*torch.sin(22*torch.pi*torch.cos(output[0, 0])**2)**5
class ex1():
    """
    示例问题1（2目标优化问题）
    
    这是一个简单的2目标优化问题，具有以下特点：
    - 决策变量：x ∈ [0, 1]
    - 目标函数1：f1(x) = x
    - 目标函数2：f2(x) = (x-1)²
    - Pareto前沿：连续曲线，x从0到1
    """
    
    def __init__(self):
        """初始化ex1问题"""
        self.num = 1000  # 用于生成Pareto前沿的采样点数量
    
    def create_pf(self):
        """
        创建ex1的真实Pareto前沿
        
        通过在决策变量空间[0,1]上均匀采样，计算对应的目标函数值。
        
        Returns:
            np.ndarray: 真实Pareto前沿点集，形状为(num, 2)
        """
        ps = np.linspace(0, 1, num=self.num)  # 在[0,1]上均匀采样
        pf = []
        for x1 in ps:
            x = torch.Tensor([[x1]])
            f = np.stack([self.f_1(x).item(), self.f_2(x).item()])
            pf.append(f)   
        pf = np.array(pf)
        return pf
    
    def f_1(self, output):
        """
        第一个目标函数：f1(x) = x
        
        Args:
            output (torch.Tensor): 决策变量，形状为(1, 1)
            
        Returns:
            torch.Tensor: 第一个目标函数值
        """
        return output[0][0]
    
    def f_2(self, output):
        """
        第二个目标函数：f2(x) = (x-1)²
        
        Args:
            output (torch.Tensor): 决策变量，形状为(1, 1)
            
        Returns:
            torch.Tensor: 第二个目标函数值
        """
        return (output[0][0] - 1) ** 2
class ex2():
    def __init__(self):
        self.num = 1000
    def create_pf(self):
        ps = np.linspace(0,5,num = self.num)
        pf = []
        for x1 in ps:
            x = torch.Tensor([[x1,x1]])
            f= np.stack([self.f_1(x).item(),self.f_2(x).item()])
            pf.append(f)   
        pf = np.array(pf)
        return pf
    def f_1(self, output):
        return (1/50)*(output[0][0]**2 + output[0][1]**2)
    def f_2(self, output):
        return (1/50)*((output[0][0]-5)**2 + (output[0][1]-5)**2)
class ex3():
    def __init__(self):
        self.num = 50
    def create_pf(self):
        u = np.linspace(0, 1, endpoint=True, num=self.num)
        v = np.linspace(0, 1, endpoint=True, num=self.num)
        tmp = []
        for i in u:
            for j in v:
                if 1-i**2-j**2 >=0:
                    tmp.append([np.sqrt(1-i**2-j**2),i,j])
                    tmp.append([i,np.sqrt(1-i**2-j**2),j])
                    tmp.append([i,j,np.sqrt(1-i**2-j**2)])
        uv = np.array(tmp)
        print(f"uv.shape={uv.shape}")
        ls = []
        for x in uv:
            x = torch.Tensor([x])
            f= np.stack([self.f_1(x).item(),self.f_2(x).item(),self.f_3(x).item()])
            ls.append(f)
        ls = np.stack(ls)
        po, pf = [], []
        for i, x in enumerate(uv):
            l_i = ls[i]
            po.append(x)
            pf.append(l_i)
        po = np.stack(po)
        pf = np.stack(pf)
        return pf
    def f_1(self, output):
        return ((output[0][0]**2 + output[0][1]**2 + output[0][2]**2+output[0][1] - 12*(output[0][2])) +12)/14

    def f_2(self, output):
        return ((output[0][0]**2 + output[0][1]**2 + output[0][2]**2\
            + 8*(output[0][0]) - 44.8*(output[0][1]) + 8*(output[0][2])) +44)/57
    def f_3(self, output):
        return ((output[0][0]**2 + output[0][1]**2 + output[0][2]**2 -44.8*(output[0][0])\
            + 8*(output[0][1]) + 8*(output[0][2]))+43.7)/56
class ex4():
    def __init__(self):
        self.num = 1000
    def create_pf(self):
        pf = [[0,0]]
        pf = np.array(pf)
        return pf
    def f_1(self, output):
        return output[0][0]
    def f_2(self, output):
        return output[0][1]
class ZDT1():
    """
    ZDT1测试函数（2目标优化问题）
    
    ZDT1是经典的多目标优化测试函数，具有以下特点：
    - 2个目标函数
    - 决策变量：x_i ∈ [0, 1], i = 1, ..., n
    - Pareto前沿：凸曲线 f2 = 1 - √f1
    - 具有可扩展的决策变量维度
    """
    
    def __init__(self):
        """初始化ZDT1问题"""
        self.n_pareto_points = 1000  # Pareto前沿采样点数量
    
    def create_pf(self):
        """
        创建ZDT1的真实Pareto前沿
        
        ZDT1的Pareto前沿为凸曲线：f2 = 1 - √f1，其中f1 ∈ [0, 1]
        
        Returns:
            np.ndarray: 真实Pareto前沿点集，形状为(n_pareto_points, 2)
        """
        x = np.linspace(0, 1, self.n_pareto_points)
        pf = np.array([x, 1 - np.sqrt(x)]).T
        return pf
    
    def f_1(self, output):
        """
        第一个目标函数：f1(x) = x1
        
        Args:
            output (torch.Tensor): 决策变量，形状为(1, n_vars)
            
        Returns:
            torch.Tensor: 第一个目标函数值
        """
        return output[0][0]
    
    def f_2(self, output):
        """
        第二个目标函数：f2(x) = g(x) * (1 - √(f1/g))
        其中 g(x) = 1 + 9/(n-1) * Σ(x_i), i=2,...,n
        
        Args:
            output (torch.Tensor): 决策变量，形状为(1, n_vars)
            
        Returns:
            torch.Tensor: 第二个目标函数值
        """
        dim = output.shape[1]
        tmp = 0
        for i in range(1, dim):
            tmp += output[0][i]
        g = 1 + (9 / (dim - 1)) * tmp
        f1 = output[0][0]
        return g * (1 - torch.sqrt(f1 / g))
class ZDT2():
    def __init__(self):
        self.n_pareto_points = 1000
    def create_pf(self):
        x = np.linspace(0, 1, self.n_pareto_points)
        pf = np.array([x, 1 - (x)**2]).T
        return pf
    def f_1(self, output):
        return output[0][0]
    def f_2(self, output):
        dim = output.shape[1]
        tmp = 0
        for i in range(1,dim):
            tmp += output[0][i]
        g = 1 + (9/(dim-1))*tmp
        f1 = output[0][0]
        return g*(1 - (f1/g)**2)

class Problem():
    """
    统一的多目标优化问题接口
    
    该类作为所有多目标优化问题的统一接口，提供标准化的访问方式。
    支持的问题类型：
    - 示例问题：ex1, ex2, ex3, ex4
    - 经典测试函数：ZDT1, ZDT2, DTLZ2
    - 测试问题：test
    """
    
    def __init__(self, name, mode):
        """
        初始化问题实例
        
        Args:
            name (str): 问题名称
            mode (str): 维度模式 ('2d' 或 '3d')
        """
        self.name = name
        self.mode = mode
        
        # 根据问题名称创建对应的问题实例
        if self.name == 'ex1':
            self.pb = ex1()
        elif self.name == 'ex2':
            self.pb = ex2()
        elif self.name == 'ex3':
            self.pb = ex3()
        elif self.name == 'ex4':
            self.pb = ex4()
        elif self.name == 'ZDT1':
            self.pb = ZDT1()
        elif self.name == 'ZDT2':
            self.pb = ZDT2()
        elif self.name == 'DTLZ2':
            self.pb = DTLZ2()
        elif self.name == 'test':
            self.pb = test()
        else:
            raise ValueError(f"未知的问题类型: {name}")
    
    def get_pf(self):
        """
        获取真实Pareto前沿
        
        Returns:
            np.ndarray: 真实Pareto前沿点集
        """
        pf = self.pb.create_pf()
        return pf
    
    def get_values(self, output):
        """
        计算给定决策变量的目标函数值
        
        Args:
            output (torch.Tensor): 决策变量
            
        Returns:
            list: 目标函数值列表
        """
        if self.mode == '2d':
            # 2目标优化
            f1, f2 = self.pb.f_1(output), self.pb.f_2(output)
            objectives = [f1, f2]
        else:
            # 3目标优化
            f1, f2, f3 = self.pb.f_1(output), self.pb.f_2(output), self.pb.f_3(output)
            objectives = [f1, f2, f3]
        return objectives

