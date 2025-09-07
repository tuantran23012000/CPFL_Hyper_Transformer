import sys
import os
sys.path.append(os.getcwd())
import time
from tqdm import tqdm
import numpy as np
import torch
from matplotlib import pyplot as plt
from utils import set_seed,visualize_3d,visualize_2d
import random
import torch.nn.functional as F
import torch
from model import Hypernet_trans,Hypernet_trans2
import argparse
from itertools import product
from problems.get_problem import Problem
from pymoo.indicators.hv import HV
import yaml
from predict import predict_result
import sys
sys.path.append('../Connected_PF')
from ..Connected_PF.tools.scalarization_function import CS_functions, EPOSolver


def sample_config(search_space_dict, reset_random_seed=False, seed=0):
    if reset_random_seed:
        random.seed(seed)
    
    config = dict()
    
    for key, value in search_space_dict.items():
        config[key] = random.choice(value)
        
    return config
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(device, cfg, criterion, pb,pf,join_input):
    model_type = "trans"
    set_seed(42)
    name = cfg['NAME']
    mode = cfg['MODE']
    ray_hidden_dim = cfg['TRAIN']['Ray_hidden_dim']
    out_dim = cfg['TRAIN']['Out_dim']
    n_tasks = cfg['TRAIN']['N_task']
    num_hidden_layer = cfg['TRAIN']['Solver'][criterion]['Num_hidden_layer']
    last_activation = cfg['TRAIN']['Solver'][criterion]['Last_activation']
    ref_point = tuple(map(int, cfg['TRAIN']['Ref_point'].split(',')))
    lr = cfg['TRAIN']['OPTIMIZER']['Lr']
    wd = cfg['TRAIN']['OPTIMIZER']['WEIGHT_DECAY']
    type_opt = cfg['TRAIN']['OPTIMIZER']['TYPE']
    epochs = cfg['TRAIN']['Epoch']
    alpha_r = cfg['TRAIN']['Alpha']
    start = 0.
    if name == 'ZDT3':
        c_s = [[0,0.8],[0.14,0.58],[0.33,0.4],[0.56,0.25],[0.8,0.1]] #ZDT3
        idxs = [0,1,2,3,4]
        expert_dim=[30,20,20,10,50]
        lrs = [0.001,0.001,0.001,0.001,0.005]
    elif name == 'ZDT3_variant':
        c_s = [[0.8,0.62],[0.01,0.7]] #ZDT3_variant
        idxs = [0,1]
        expert_dim=[10,10]
        lrs = [0.001,0.005]
    elif name == 'DTLZ7':
        c_s = [[0.6,0.6,0.4],[0.01,0.5,0.5],[0.01,0.01,0.7],[0.5,0.01,0.6]]
        expert_dim=[30,30,30,30]
        lrs = [0.003,0.001,0.005,0.001]
        idxs = [0,1,2,3]
    if joint_input:
        hnet = Hypernet_trans2(ray_hidden_dim=ray_hidden_dim, out_dim=out_dim, target_hidden_dim=ray_hidden_dim, n_hidden=1, n_tasks=n_tasks)
    else:
        hnet = Hypernet_trans(ray_hidden_dim=ray_hidden_dim, out_dim=out_dim, expert_dim=expert_dim, n_experts=len(expert_dim), n_tasks=n_tasks)
    hnet = hnet.to(device)
    param = count_parameters(hnet)
    print(hnet)
    print("Model size: ",param)
    print("Model type: ",model_type)
    sol = []
    if type_opt == 'adam':
        optimizer = torch.optim.Adam(hnet.parameters(), lr = lr, weight_decay=wd) 
    elif type_opt == 'adamw':
        optimizer = torch.optim.AdamW(hnet.parameters(), lr = lr, weight_decay=wd)
    start = time.time()

    for epoch in tqdm(range(epochs)):
        idx = random.choice(idxs)
        c_ = c_s[idx]
        l_r = lrs[idx]
        optimizer.param_groups[0]['lr'] = l_r
        c = torch.tensor(c_)
        if n_tasks == 2:
            u1 = random.uniform(c[0], 1)
            u2 = random.uniform(c[1], 1)
            u = np.array([u1,u2])
            lda = (u/np.linalg.norm(u,1))
        else:
            u1 = random.uniform(c[0], 1)
            u2 = random.uniform(c[1], 1)
            u3 = random.uniform(c[2], 1)
            u = np.array([u1,u2,u3])
            lda = (u/np.linalg.norm(u,1))
        ray = torch.from_numpy(lda).float()
        hnet.train()
        optimizer.zero_grad()
        if joint_input:
            output = hnet(ray,c)
        else:
            output = hnet(ray,idx)

        if cfg["NAME"] == "ex3":
            output = F.softmax(output,dim=2)
            output = torch.sqrt(output)
        else:
            output = F.sigmoid(output)
        if cfg["NAME"] == "ex2":
            output = 5*output
                
        ray_cs = 1/ray
        ray = ray.squeeze(0)
        obj_values = []
        objectives = pb.get_values(output)
        for i in range(len(objectives)):
            obj_values.append(objectives[i])
        losses = torch.stack(obj_values)
        
        # 根据solver类型选择相应的标量化函数计算损失
        CS_func = CS_functions(losses, ray)
        
        if criterion == 'LS':
            # 线性标量化
            loss = CS_func.linear_function()
        elif criterion == 'KL':
            # KL散度标量化
            loss = CS_func.KL_function()
        elif criterion == 'Cheby':
            # 切比雪夫标量化（带约束）
            loss = CS_func.chebyshev_function(c)
        elif criterion == 'Utility':
            # 效用函数标量化 - 需要从配置中获取Ub参数
            ub = cfg['TRAIN']['Solver'][criterion]['Ub'] if 'Ub' in cfg['TRAIN']['Solver'][criterion] else 8
            loss = CS_func.utility_function(ub)
        elif criterion == 'Cosine':
            # 余弦相似度标量化
            loss = CS_func.cosine_function()
        elif criterion == 'Cauchy':
            # 柯西-施瓦茨标量化
            loss = CS_func.cauchy_schwarz_function()
        elif criterion == 'Prod':
            # 乘积标量化
            loss = CS_func.product_function()
        elif criterion == 'Log':
            # 对数标量化
            loss = CS_func.log_function()
        elif criterion == 'AC':
            # 增强切比雪夫标量化
            rho = cfg['TRAIN']['Solver'][criterion]['Rho'] if 'Rho' in cfg['TRAIN']['Solver'][criterion] else 0.1
            loss = CS_func.ac_function(rho)
        elif criterion == 'MC':
            # 修改切比雪夫标量化
            rho = cfg['TRAIN']['Solver'][criterion]['Rho'] if 'Rho' in cfg['TRAIN']['Solver'][criterion] else 0.1
            loss = CS_func.mc_function(rho)
        elif criterion == 'HV':
            # 超体积标量化
            rho = cfg['TRAIN']['Solver'][criterion]['Rho'] if 'Rho' in cfg['TRAIN']['Solver'][criterion] else 100
            dynamic_weight = ray
            loss = CS_func.hv_function(dynamic_weight, rho)
        elif criterion == 'EPO':
            # EPO需要特殊处理，使用EPOSolver
            n_params = sum(p.numel() for p in hnet.parameters() if p.requires_grad)
            epo_solver = EPOSolver(n_tasks=n_tasks, n_params=n_params)
            loss = epo_solver.get_weighted_loss(losses, ray, list(hnet.parameters()))
        elif criterion == 'CPMTL':
            # CPMTL使用线性标量化作为fallback
            loss = CS_func.linear_function()
        elif criterion == 'HVI':
            # HVI (Hypervolume Indicator)
            rho = cfg['TRAIN']['Solver'][criterion]['Rho'] if 'Rho' in cfg['TRAIN']['Solver'][criterion] else 200
            dynamic_weight = ray
            loss = CS_func.hv_function(dynamic_weight, rho)
        else:
            # 默认使用原来的简化切比雪夫标量化
            print(f"警告: 未识别的solver类型 {criterion}，使用默认的切比雪夫标量化")
            loss = max(torch.abs(losses-c) * ray)
        loss.backward()

        optimizer.step()
        tmp = []
        for i in range(len(objectives)):
            tmp.append(objectives[i].cpu().detach().numpy().tolist())
        if epoch >1000:
            sol.append(tmp)

        # sol.append(ray.cpu().detach().numpy().tolist())

    end = time.time()
    time_training = end-start
    ind = HV(ref_point=np.ones(n_tasks))
    print(np.array(sol).shape,pf.shape)
    hv_loss_app = ind(np.array(sol))
    hv_loss = ind(pf)
    print("HV approximate: ",hv_loss_app)
    print("HV: ",hv_loss)
    print("SUb: ",hv_loss - hv_loss_app)
    if n_tasks == 3:
        visualize_3d(sol,pf,cfg,criterion,pb,model_type)
    else:
        visualize_2d(sol,pf,cfg,criterion,pb,model_type)
    # torch.save(hnet,"test.pt")
    # igd, targets_epo, results1, contexts,med,c,model_type = predict_result(device,cfg,criterion,pb,pf,join_input,all_subnets,hnet,num_e=None,contexts = [])
    # from utils import vis_3d
    # vis_3d(cfg,targets_epo, results1, contexts,pb,pf,criterion,igd,med,model_type,join_input)
if __name__ == "__main__":
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver", type=str, choices=["LS", "KL","Cheby","Utility","Cosine","Cauchy","Prod","Log","AC","MC","HV","CPMTL","EPO","HVI"],
        default="Cheby", help="solver"
    )
    parser.add_argument(
        "--problem", type=str, choices=["ZDT3","DTLZ7","ZDT3_variant"],
        default="ZDT3_variant", help="solver"
    )
    parser.add_argument(
        "--mode", type=str,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        default="test"
    )
    parser.add_argument(
        "--joint_input", type=bool,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        default=False
    )
    args = parser.parse_args()
    criterion = args.solver 
    joint_input = args.joint_input
    print("joint_input: ",joint_input)
    print("Scalar funtion: ",criterion)
    problem = args.problem
    config_file = "./configs/"+str(problem)+".yaml"
    with open(config_file) as stream:
        cfg = yaml.safe_load(stream)
    pb = Problem(problem, cfg['MODE'])
    pf = pb.get_pf()

    train_epoch(device,cfg,criterion,pb,pf,joint_input)