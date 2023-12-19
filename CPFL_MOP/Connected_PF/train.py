import sys
import os
sys.path.append(os.getcwd())
import time
from tqdm import tqdm
import numpy as np
import torch
from tools.scalarization_function import CS_functions,EPOSolver
from tools.hv import HvMaximization
from models import Hypernet_mlp, Hypernet_trans
from tools.utils import set_seed
import random
import torch.nn.functional as F
import torch
from predict import predict_result
from itertools import product

def sample_config(search_space_dict, reset_random_seed=False, seed=0):
    if reset_random_seed:
        random.seed(seed)
    
    config = dict()
    
    for key, value in search_space_dict.items():
        config[key] = random.choice(value)
        
    return config
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def train_epoch(device, cfg, criterion, pb,pf,model_type):
    # model_type = "trans"
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
    if name == 'ex1':
        c_s = [[0,0.8],[0.1,0.6],[0.2,0.4],[0.35,0.22],[0.6,0.1]] #CVX1
    elif name == 'ex2':
        c_s = [[0,0.6],[0.02,0.4],[0.16,0.2],[0.2,0.15],[0.4,0.02]] #CVX2
    elif name == 'ex3':
        c_s = [[0.15,0.2,0.7],[0.2,0.5,0.6],[0.2,0.7,0.4],[0.35,0.6,0.22],[0.6,0.1,0.46]] #CVX3
    elif name == 'ZDT1':
        c_s = [[0,0.8],[0.1,0.6],[0.2,0.4],[0.35,0.22],[0.6,0.1]] #ZDT1
    elif name == 'ZDT2':
        c_s = [[0.1,0.9],[0.1,0.6],[0.2,0.4],[0.35,0.22],[0.6,0.1]] #ZDT2
    elif name == 'DTLZ2':
        c_s = [[0.15,0.2,0.7],[0.2,0.5,0.6],[0.2,0.7,0.4],[0.35,0.6,0.22],[0.6,0.1,0.46]] #DTLZ2
    
    
    sol = []
    
    for c_ in c_s:
        if model_type == "mlp":
            hnet = Hypernet_mlp(ray_hidden_dim=ray_hidden_dim, out_dim=out_dim, target_hidden_dim=ray_hidden_dim, n_hidden=1, n_tasks=n_tasks)
        else:
            hnet = Hypernet_trans(ray_hidden_dim=ray_hidden_dim, out_dim=out_dim, target_hidden_dim=ray_hidden_dim, n_hidden=1, n_tasks=n_tasks)
        hnet = hnet.to(device)
        param = count_parameters(hnet)
        if type_opt == 'adam':
            optimizer = torch.optim.Adam(hnet.parameters(), lr = lr, weight_decay=wd) 
        elif type_opt == 'adamw':
            optimizer = torch.optim.AdamW(hnet.parameters(), lr = lr, weight_decay=wd)
        for epoch in tqdm(range(epochs)):
            
            # print("Model size: ",param)
            # print("Model type: ",model_type)
            c = torch.tensor(c_)
            if n_tasks == 2:
                u1 = random.uniform(c[0], 1)
                u2 = random.uniform(c[1], 1)
                u = np.array([u1,u2])
            else:
                u1 = random.uniform(c[0], 1)
                u2 = random.uniform(c[1], 1)
                u3 = random.uniform(c[2], 1)
                u = np.array([u1,u2,u3])
            lda = (u/np.linalg.norm(u,1))
            ray = torch.from_numpy(lda).float()
            hnet.train()
            optimizer.zero_grad()

            output = hnet(ray)
            if model_type == 'trans':
                if cfg["NAME"] == "ex3":
                    
                    output = F.softmax(output,dim=1)
                else:
                    output = F.sigmoid(output)   
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

            ray_cs = 1/ray
            ray = ray.squeeze(0)
            obj_values = []
            objectives = pb.get_values(output)
            for i in range(len(objectives)):
                obj_values.append(objectives[i])
            losses = torch.stack(obj_values)
            CS_func = CS_functions(losses,ray)
            loss = CS_func.chebyshev_function(c)
            loss.backward()
            optimizer.step()
            tmp = []
            for i in range(len(objectives)):
                tmp.append(objectives[i].cpu().detach().numpy().tolist())
            sol.append(tmp)

        torch.save(hnet,("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_" + str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+".pt"))
    return sol


