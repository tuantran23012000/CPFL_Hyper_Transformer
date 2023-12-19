import torch
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import random
import argparse
from tools.utils import find_target, circle_points_random, get_d_paretomtl
from tools.utils import circle_points, sample_vec
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

# from models.hyper_toy import Toy_Hypernetwork, Toy_Targetnetwork
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def get_rays(cfg,num_ray_init):
    contexts = np.array(sample_vec(cfg['TRAIN']['N_task'],num_ray_init))
    tmp = []
    for r in contexts:
        flag = True
        for i in r:
            if i <=0.16:
                flag = False
                break
        if flag:

            tmp.append(r)
    contexts = np.array(tmp)
    return contexts
def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = 1e-6 if min_angle is None else min_angle
    ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]
def simplex(n_vals):
    base = np.linspace(0, 0.25, n_vals, endpoint=False)
    coords = np.asarray(list(itertools.product(base, repeat=3)))
    return coords[np.isclose(coords.sum(axis=-1), 0.25)]
def predict_result(device,cfg,criterion,pb,pf,model_type):
    
    # model_type = "trans"
    print("Model type: ",model_type)
    mode = cfg['MODE']
    name = cfg['NAME']   
    print("Problem: ",name) 
    num_ray_init = cfg['EVAL']['Num_ray_init']
    num_ray_test = cfg['EVAL']['Num_ray_test']
    out_dim = cfg['TRAIN']['Out_dim']
    ray_hidden_dim = cfg['TRAIN']['Ray_hidden_dim']
    n_tasks = cfg['TRAIN']['N_task']
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
    count = 0
    for se in tqdm(range(30)):
        count += 1
        set_seed(se)
        meds_se = []
        for c_ in c_s:
            hnet = torch.load("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_" + str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+".pt",map_location=device)
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
            
            # np.save('./predict/target_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'.npy',targets_epo)
            # np.save('./predict/predict_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'.npy',results1)
            # np.save('./predict/med_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'.npy',med)
            # #np.save('./predict/ray_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'.npy',contexts)
            meds_se.append(np.mean(np.array(meds_c).tolist()))
    print("Mean: ",np.mean(np.array(meds_se)))
    print("Std: ",np.std(np.array(meds_se)))
