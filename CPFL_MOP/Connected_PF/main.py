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

def run_train(cfg,criterion,device,problem,model_type):
    pb = Problem(problem, cfg['MODE'])
    pf = pb.get_pf()
    if cfg['MODE'] == '2d':
        sol = train_epoch(device,cfg,criterion,pb,pf,model_type)
    else:
        sol = train_epoch(device,cfg,criterion,pb,pf,model_type)
def run_predict(cfg,criterion,device,problem,model_type):
    pb = Problem(problem, cfg['MODE'])
    pf = pb.get_pf()
    if cfg['MODE'] == '2d':   
        predict_result(device,cfg,criterion,pb,pf,model_type)
    else:    
        predict_result(device,cfg,criterion,pb,pf,model_type)

if __name__ == "__main__":
    device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver", type=str, choices=["LS", "KL","Cheby","Utility","Cosine","Cauchy","Prod","Log","AC","MC","HV","CPMTL","EPO","HVI"],
        default="Cheby", help="solver"
    )
    parser.add_argument(
        "--problem", type=str, choices=["ex1", "ex2","ex3","ex4","ZDT1","ZDT2","DTLZ2"],
        default="ex1", help="solver"
    )
    parser.add_argument(
        "--mode", type=str,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        default="test"
    )
    parser.add_argument(
        "--model_type", type=str,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        default="mlp"
    )
    args = parser.parse_args()
    criterion = args.solver 
    model_type = args.model_type
    print("model_type: ",model_type)
    print("Scalar funtion: ",criterion)
    problem = args.problem
    config_file = "./configs/"+str(problem)+".yaml"
    with open(config_file) as stream:
        cfg = yaml.safe_load(stream)
    if args.mode == "train":
        run_train(cfg,criterion,device,problem,model_type)
    else:
        run_predict(cfg,criterion,device,problem,model_type)