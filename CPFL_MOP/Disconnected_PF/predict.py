import torch
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import random
import argparse
from utils import find_target, circle_points_random, get_d_paretomtl
from utils import circle_points, sample_vec
from metrics import IGD, MED
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
def predict_result1(device,cfg,criterion,pb,pf,join_input,num_e=None,contexts = []):
    from tools.utils import set_seed
    model_type = "trans"
    print("Model type: ",model_type)

    se = 7 #ex1:32-3rays, ex2:25-3rays, ex3:14-4rays, ZDT1:17-3rays, ZDT2:4-3rays, DTLZ2:12-4rays  
    set_seed(se)
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
    elif name == 'ZDT3':
        c_s = [[0,0.8],[0.14,0.58],[0.33,0.4],[0.56,0.25],[0.8,0.1]] #ZDT3
        #c_s = [[0.8,0.1],[0,0.82],[0.17,0.6],[0.4,0.42],[0.6,0.27]] #ZDT3
        c_s = [[0.8,0.1]] 
    elif name == 'ZDT3_variant':
        c_s = [[0,0.6],[0.79,0.62]] #ZDT3_variant
    elif name == 'DTLZ2':
        c_s = [[0.15,0.2,0.7],[0.2,0.5,0.6],[0.2,0.7,0.4],[0.35,0.6,0.22],[0.6,0.1,0.46]] #DTLZ2
    elif name == 'DTLZ7':
        c_s = [[0.1,0.7,0.7],[0.1,0.1,0.83],[0.77,0.7,0.5],[0.7,0.1,0.65]] #DTLZ7
    #c_s = [[0,0.8],[0.14,0.58],[0.33,0.4],[0.56,0.25],[0.8,0.1]] #ZDT3
    #c_s = [[0,0.8],[0.1,0.6],[0.2,0.4],[0.35,0.22],[0.6,0.1]] #CVX1
    #c_s = [[0,0.6],[0.02,0.4],[0.16,0.2],[0.2,0.15],[0.4,0.02]] #CVX2
    #c_s = [[0,0.8],[0.1,0.6],[0.2,0.4],[0.35,0.22],[0.6,0.1]] #ZDT1
    #c_s = [[0.1,0.9],[0.1,0.6],[0.2,0.4],[0.35,0.22],[0.6,0.1]] #ZDT2
    #c_s = [[0.15,0.2,0.7],[0.2,0.5,0.6],[0.2,0.7,0.4],[0.35,0.6,0.22],[0.6,0.1,0.46]] #CVX3
    #c_s = [[0.1,0.7,0.7],[0.1,0.1,0.83],[0.77,0.7,0.5],[0.7,0.1,0.65]] #DTLZ7
    #c_s = [[0,0.6],[0.79,0.62]] #ZDT3_variant
    #c_s = [[0.15,0.2,0.7],[0.2,0.5,0.6],[0.2,0.7,0.4],[0.35,0.6,0.22],[0.6,0.1,0.46]] #DTLZ2
    # meds_se = []
    # param = 0
    # for se in tqdm(range(30)):
    #     set_seed(se)
    #     meds_c = []
    # hnet1 = torch.load("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_"+ str(model_type)+"_join.pt",map_location=device)

    c_ = [0.8,0.1]
    if join_input:
        hnet1 = torch.load("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_"+ str(model_type)+"_join.pt",map_location=device)
    else:
        hnet1 = torch.load("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_" + str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+".pt",map_location=device)
        #hnet1 = torch.load("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_" + str(model_type)+"_test.pt",map_location=device)
    print(hnet1)
    hnet1.eval()
    # net = torch.load("./save_weights/best_weight_net_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_"+ str(cfg['TRAIN']['Ray_hidden_dim'])+".pt",map_location=device)
    # net.eval()

    hnet1 = hnet1.to(device)
    # if se == 0:
    #     if join_input:
    #         param = count_parameters(hnet1)   
    #     else:
    #         param += count_parameters(hnet1)
    # elif se == 1:
    #     print("Model size: ",param)
    results1 = []
    targets_epo = []
    contexts = get_rays(cfg, num_ray_init)
    rng = np.random.default_rng()
    contexts = rng.choice(contexts,num_ray_test)
    #contexts = np.array([[0.2, 0.5,0.3], [0.4, 0.25,0.35],[0.3,0.2,0.5],[0.55,0.2,0.25]])
    #contexts = np.array([[0.2, 0.5,0.3], [0.4, 0.25,0.35],[0.3,0.2,0.5],[0.55,0.2,0.25]])
    min_angle = 0.1
    max_angle = np.pi / 2 - 0.1
    num_rays = 25
    contexts = circle_points(num_rays, min_angle=min_angle, max_angle=max_angle)
    contexts = np.array([[0.2, 0.5,0.3], [0.4, 0.25,0.35],[0.3,0.2,0.5],[0.55,0.2,0.25]])
    contexts = np.array([[0.2, 0.8], [0.4, 0.6],[0.7,0.3]]) 
    contexts = np.array([[0.43412827265038134, 0.5658717273496187], [0.6131066337394924, 0.3868933662605077], [0.5557075300602374, 0.4442924699397625]]) 
    # c = np.array([0.2,0.5,0.4])
    
    if n_tasks == 2:
        #c = np.array([0,0.7])
        c = np.array(c_)
    else:
        c = np.array(c_)
    tmp = []


    #for r in contexts:
    #for se in range(30):

    results1 = []
    targets_epo = []
    for i in range(3):
        
        # r = r/r.sum()
        # #print(r)
        # r_inv = 1. / r
        # ray = torch.Tensor(r.tolist()).to(device)
        
        c_in = torch.Tensor(c.tolist()).to(device)
        #output = hnet1(ray)
        #output = torch.sqrt(output)
        if n_tasks == 2:
            u1 = random.uniform(c[0], 1)
            u2 = random.uniform(c[1], 1)
            u = np.array([u1,u2])
        else:
            u1 = random.uniform(c[0], 1)
            u2 = random.uniform(c[1], 1)
            u3 = random.uniform(c[2], 1)
            u = np.array([u1,u2,u3])
        
        r = u/np.linalg.norm(u,1)
        tmp.append(r)
        ray = torch.from_numpy(r).float()

        # weights = hnet1(ray)
        # output = net(weights,ray)
        # output = hnet1(ray,c_in)
        if join_input:
            output = hnet1(ray,c_in)
        else:
            output = hnet1(ray)
        if model_type == 'trans':
            if cfg["NAME"] == "ex3":
                output = F.softmax(output,dim=2)
            else:
                output = F.sigmoid(output)
            #output = torch.mean(output,dim=0)  
            #output = output.unsqueeze(0) 
            #print(output)
        else:
            if cfg["NAME"] == "ex3":
                output = F.softmax(output)
            else:
                output = F.sigmoid(output)
            output = output.unsqueeze(0)
        if cfg["NAME"] == "ex2":
            output = 5*output
        elif cfg["NAME"] == "ex3":
            
            output = torch.sqrt(output)

        #output = torch.sqrt(output)
        objectives = pb.get_values(output)
        obj_values = []
        
        for j in range(len(objectives)):
            obj_values.append(objectives[j].cpu().detach().numpy().tolist())
        results1.append(obj_values)
        # if criterion == "Cauchy":
        #     target_epo = find_target(pf, criterion = criterion, context = r_inv.tolist(),c=c,cfg=cfg)
        # else:
        target_epo = find_target(pf, criterion = criterion, context = r.tolist(),c=c,cfg=cfg)
        targets_epo.append(target_epo)
    targets_epo = np.array(targets_epo)

    # print(targets_epo)
    # print(results1)
    results1 = np.array(results1, dtype='float32')
    med = np.mean(np.sqrt(np.sum(np.square(targets_epo-results1),axis = 1)))
    med = MED(targets_epo, results1)
    #meds_c.append(med)
    print(med)
    d_i = []
    for target in pf:
        d_i.append(np.min(np.sqrt(np.sum(np.square(target-results1),axis = 1))))
    igd = np.mean(np.array(d_i))
    
    igd = IGD(pf, results1)
    # tmp.append(med.tolist())
    # print(np.mean(np.array(tmp)))
    # print(np.std(np.array(tmp)))
    contexts = np.array(tmp)
    if join_input:
        np.save('./predict/target_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'_join.npy',targets_epo)
        np.save('./predict/predict_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'_join.npy',results1)
        np.save('./predict/med_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'_join.npy',med)
        np.save('./predict/ray_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'_join.npy',contexts)
    else:
        np.save('./predict/target_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'.npy',targets_epo)
        np.save('./predict/predict_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'.npy',results1)
        np.save('./predict/med_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'.npy',med)
        np.save('./predict/ray_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'.npy',contexts)
    #     meds_se.append(np.mean(np.array(meds_c).tolist()))
    # print("Mean: ",np.mean(np.array(meds_se)))
    # print("Std: ",np.std(np.array(meds_se)))
    return igd, targets_epo, results1, contexts, med,c,model_type

def predict_result(device,cfg,criterion,pb,pf,join_input,all_subnets,hnet1,num_e=None,contexts = []):
    from utils import set_seed
    model_type = "trans"
    print("Model type: ",model_type)

    # se = 21 #DTLZ7: 11 #ex1:32-3rays, ex2:25-3rays, ex3:14-4rays, ZDT1:17-3rays, ZDT2:4-3rays, DTLZ2:12-4rays  
    # set_seed(se)
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
    elif name == 'ZDT3':
        c_s = [[0,0.8],[0.14,0.58],[0.33,0.4],[0.56,0.25],[0.8,0.1]] #ZDT3
        c_s = [[0.8,0.1],[0.56,0.25],[0,0.8],[0.14,0.58],[0.33,0.4]] #ZDT3
        #c_s = [[0.8,0.1]] 
    elif name == 'ZDT3_variant':
        c_s = [[0.79,0.62],[0,0.6]] #ZDT3_variant
    elif name == 'DTLZ2':
        c_s = [[0.15,0.2,0.7],[0.2,0.5,0.6],[0.2,0.7,0.4],[0.35,0.6,0.22],[0.6,0.1,0.46]] #DTLZ2
    elif name == 'DTLZ7':
        c_s = [[0.6,0.6,0.4],[0.01,0.5,0.5],[0.01,0.01,0.7],[0.5,0.01,0.6]]
 
    #hnet1 = torch.load("./save_weights/test.pt",map_location=device)
        #hnet1 = torch.load("./save_weights/best_weight_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_" + str(model_type)+"_test.pt",map_location=device)
    #print(hnet1)
    
    meds_se = []
    param = 0
    for se in tqdm(range(30)):
        set_seed(se)
        meds_c = []
        idx = -1
        for c_,cfg_ in zip(c_s,all_subnets):
            idx += 1
            # if cfg_['layer1'] == 10:
            #     c = np.array([0.79,0.62]) #[[0.79,0.62],[0,0.6]]
            # else:
            #     #print(cfg_)
            #     c = np.array([0,0.6])

            
            hnet1.eval()
            #hnet1.set_sample_config(cfg_)
            #print(hnet1)
            # net = torch.load("./save_weights/best_weight_net_"+str(criterion)+"_"+str(mode)+"_"+str(name)+"_"+ str(cfg['TRAIN']['Ray_hidden_dim'])+".pt",map_location=device)
            # net.eval()

            hnet1 = hnet1.to(device)
            # if se == 0:
            #     if join_input:
            #         param = count_parameters(hnet1)   
            #     else:
            #         param += count_parameters(hnet1)
            # elif se == 1:
            #     print("Model size: ",param)
            results1 = []
            targets_epo = []
            contexts = get_rays(cfg, num_ray_init)
            rng = np.random.default_rng()
            contexts = rng.choice(contexts,num_ray_test)
            #contexts = np.array([[0.2, 0.5,0.3], [0.4, 0.25,0.35],[0.3,0.2,0.5],[0.55,0.2,0.25]])
            #contexts = np.array([[0.2, 0.5,0.3], [0.4, 0.25,0.35],[0.3,0.2,0.5],[0.55,0.2,0.25]])
            min_angle = 0.1
            max_angle = np.pi / 2 - 0.1
            num_rays = 25
            contexts = circle_points(num_rays, min_angle=min_angle, max_angle=max_angle)
            contexts = np.array([[0.2, 0.5,0.3], [0.4, 0.25,0.35],[0.3,0.2,0.5],[0.55,0.2,0.25]])
            contexts = np.array([[0.2, 0.8], [0.4, 0.6],[0.7,0.3]]) 
            contexts = np.array([[0.43412827265038134, 0.5658717273496187], [0.6131066337394924, 0.3868933662605077], [0.5557075300602374, 0.4442924699397625]]) 
            # c = np.array([0.2,0.5,0.4])
            
            if n_tasks == 2:
                #c = np.array([0,0.7])
                c = np.array(c_)
            else:
                c = np.array(c_)

            tmp = []


            #for r in contexts:
            #for se in range(30):

            results1 = []
            targets_epo = []
            for i in range(3):
                
                # r = r/r.sum()
                # #print(r)
                # r_inv = 1. / r
                # ray = torch.Tensor(r.tolist()).to(device)
                
                c_in = torch.Tensor(c.tolist()).to(device)
                #output = hnet1(ray)
                #output = torch.sqrt(output)
                if n_tasks == 2:
                    u1 = random.uniform(c[0], 1)
                    u2 = random.uniform(c[1], 1)
                    u = np.array([u1,u2])
                else:
                    u1 = random.uniform(c[0], 1)
                    u2 = random.uniform(c[1], 1)
                    u3 = random.uniform(c[2], 1)
                    u = np.array([u1,u2,u3])
                    lda = 1/(u/np.linalg.norm(u,1))
                    #ray = 
                    # if count == 0:
                    #     lda = (u/np.linalg.norm(u,1))
                    # else:
                    #     lda = (u/np.linalg.norm(u,1))
                    #     u = np.array([1/u1,1/u2,1/u3])
                    # else:
                    #     u = np.array([u1,u2,u3]) 
                r = lda/np.linalg.norm(lda,1)
                ray = torch.from_numpy(r).float()
                
                #r = u/np.linalg.norm(u,1)
                tmp.append(r)
                #ray = torch.from_numpy(r).float()

                # weights = hnet1(ray)
                # output = net(weights,ray)
                # output = hnet1(ray,c_in)
                if join_input:
                    output = hnet1(ray,c_in)
                else:
                    #print(ray)
                    output = hnet1(ray,c_in)
                    #print(output)
                if model_type == 'trans':
                    if cfg["NAME"] == "ex3":
                        output = F.softmax(output,dim=2)
                    else:
                        output = F.sigmoid(output)
                    #output = torch.mean(output,dim=0)  
                    #output = output.unsqueeze(0) 
                    
                else:
                    if cfg["NAME"] == "ex3":
                        output = F.softmax(output)
                    else:
                        output = F.sigmoid(output)
                    output = output.unsqueeze(0)
                if cfg["NAME"] == "ex2":
                    output = 5*output
                elif cfg["NAME"] == "ex3":
                    
                    output = torch.sqrt(output)

                #output = torch.sqrt(output)
                objectives = pb.get_values(output)
                obj_values = []
                
                for j in range(len(objectives)):
                    obj_values.append(objectives[j].cpu().detach().numpy().tolist())
                results1.append(obj_values)
                # if criterion == "Cauchy":
                #     target_epo = find_target(pf, criterion = criterion, context = r_inv.tolist(),c=c,cfg=cfg)
                # else:
                target_epo = find_target(pf, criterion = criterion, context = r.tolist(),c=c,cfg=cfg)
                targets_epo.append(target_epo)
            targets_epo = np.array(targets_epo)

            # print(targets_epo)
            # print(results1)
            results1 = np.array(results1, dtype='float32')
            med = np.mean(np.sqrt(np.sum(np.square(targets_epo-results1),axis = 1)))
            med = MED(targets_epo, results1)
            meds_c.append(med)
            #print(med)
            d_i = []
            for target in pf:
                d_i.append(np.min(np.sqrt(np.sum(np.square(target-results1),axis = 1))))
            igd = np.mean(np.array(d_i))
            
            igd = IGD(pf, results1)
            # tmp.append(med.tolist())
            # print(np.mean(np.array(tmp)))
            # print(np.std(np.array(tmp)))
            contexts = np.array(tmp)
            if join_input:
                np.save('./predict/target_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'_join.npy',targets_epo)
                np.save('./predict/predict_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'_join.npy',results1)
                np.save('./predict/med_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'_join.npy',med)
                np.save('./predict/ray_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'_join.npy',contexts)
            else:
                np.save('./predict/target_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'.npy',targets_epo)
                np.save('./predict/predict_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'.npy',results1)
                np.save('./predict/med_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'.npy',med)
                np.save('./predict/ray_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c_[0])+"_"+str(c_[1])+'.npy',contexts)
        meds_se.append(np.mean(np.array(meds_c).tolist()))
    print("Mean: ",np.mean(np.array(meds_se)))
    print("Std: ",np.std(np.array(meds_se)))
    return igd, targets_epo, results1, contexts, med,c,model_type