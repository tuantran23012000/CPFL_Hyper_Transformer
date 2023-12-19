import json
import time
import numpy as np
import torch
from tqdm import trange, tqdm
import os
print(os.getcwd())
from datasets.get_dataset import get_dataset
from losses.get_loss import get_loss
from models.get_model import get_model
from utils.utils import (
    circle_points,
    count_parameters,
    set_logger,
    set_seed,
    ReDirectSTD,
    get_test_rays,
    hypervolumn,
    count_parameters
)
from utils.solvers import EPOSolver, LinearScalarizationSolver, ChebyshevBasedSolver, UtilityBasedSolver

@torch.no_grad()
def eval_celeb(hnet, net, val_loader, rays, device,params):
    hnet.eval()
    tasks = params["tasks"]
    num_tasks = params["num_tasks"]
    loss_fn = get_loss(params)
    losses_all = []
    
    for ray in tqdm(rays):
        num_samples = 0
        losses = np.zeros(num_tasks)
        bs = 0
        ray = torch.from_numpy(
            ray.astype(np.float32).flatten()
        ).to(device)
        for k,batch in enumerate(val_loader):
            hnet.zero_grad()
            img = batch[0].to(device)
            bs = img.shape[0]
            num_samples += bs
            labels = {}
            weights = hnet(ray)
            logit = net(img, weights)
            loss_train = []
            pred_task = []
            gt = []
            for i, t in enumerate(tasks):
                labels = batch[i + 1].to(device)

                loss_train.append(loss_fn[t](logit[i],labels))
            e = torch.stack(loss_train, -1).detach().cpu().tolist()
            losses += bs*np.array(e)
        losses /= num_samples
        losses_all.append(losses)
    hv = get_performance_indicator(
        "hv",
    ref_point=np.ones(params["num_tasks"])
        )
    hv_eval = hv.do(np.array(losses_all))
    return  hv_eval
@torch.no_grad()
def eval_mnist(hnet, net, val_loader, test_ray, device,params,configs):
    hnet.eval()
    tasks = params["tasks"]
    loss_fn = get_loss(params,configs)
    losses_all = []
    
    for ray in tqdm(test_ray):
        num_samples = 0
        losses = np.zeros(params["num_tasks"])
        bs = 0
        ray = torch.from_numpy(
            ray.astype(np.float32).flatten()
        ).to(device)
        for k,batch in enumerate(val_loader):
            hnet.zero_grad()
            batch = (t.to(device) for t in batch)
            if configs["name_exp"] == "mnist":
                xs, ys = batch
            elif configs["name_exp"] == "celebA":
                xs = batch[0]
                ys = batch[1:params["num_task"]]
            
            bs = len(ys)
            num_samples += bs
            weights = hnet(ray)
            logit = net(xs, weights)
            loss_train = []
            for i, t in enumerate(tasks):
                loss_train.append(loss_fn[t](logit[i],ys[:, i]))
            e = torch.stack(loss_train, -1).detach().cpu().tolist()
            losses += bs*np.array(e)
        losses /= num_samples
        losses_all.append(losses)
    hv_eval = hypervolumn(np.array(losses_all),np.ones(params["num_tasks"]))
    return  hv_eval
def train_mnist(hnet, net, train_loader, device, params,configs, optimizer,loss_fn,solver):
    for i, batch in enumerate(train_loader):
        hnet.train()
        optimizer.zero_grad()
        if configs["name_exp"] == "mnist":
            img, ys = batch
        elif configs["name_exp"] == "celebA":
            img = batch[0]
            ys = batch[1:params["num_task"]]
        img = img.to(device)
        ys = ys.to(device)
        ray = torch.from_numpy(
            np.random.dirichlet(tuple([params["alpha"]]*params['num_tasks']), 1).astype(np.float32).flatten()
        ).to(device)

        weights = hnet(ray)
        logit = net(img, weights)
        loss_train = []
        for i, t in enumerate(params["tasks"]):
            loss_train.append(loss_fn[t](logit[i],ys[:, i]))
        losses = torch.stack(loss_train, -1)

        ray = ray.squeeze(0)
        loss = solver(losses, ray, list(hnet.parameters()))
        loss.backward()
        optimizer.step()

def train_celebA(params,configs,test_ray,model_type,device):
    for k,batch in tqdm(enumerate(train_loader)):
        hnet.train()
        optimizer.zero_grad()
        img = batch[0].to(device)
        labels = {}
        bs = img.shape[0]
        #if alpha > 0:
        ray = torch.from_numpy(
            np.random.dirichlet(tuple([params["alpha"]]*params['num_tasks']), 1).astype(np.float32).flatten()
        ).to(device)
        #print(ray.shape)
        weights = hnet(ray)
        logit = net(img, weights)
        loss_train = []
        pred_task = []
        gt = []
        for i, t in enumerate(all_tasks):
            out_vals = []
            labels = batch[i + 1].to(device)
            out_vals.append(logit[i])
            loss_train.append(loss_fn[t](logit[i],labels))
        ray = ray.squeeze(0)
        loss = solver(torch.stack(loss_train, -1), ray, list(hnet.parameters()))
        loss.backward()
        optimizer.step()
def train(params,configs,test_ray,model_type,device):
    stdout_file = "./logs/" + params['dataset']+"_" + params["solver"]+"_"+model_type+".txt"
    ReDirectSTD(stdout_file, 'stdout', False)
    # ----
    # Nets
    # ----
    print(configs)
    print(params)
    hnet, net = get_model(params,configs)
    hnet = hnet.to(device)
    net = net.to(device)
    optimizer = torch.optim.Adam(hnet.parameters(), lr=params["lr"])
    # ----
    # Datasets
    # ----
    if configs["name_exp"] == 'nlp':
        train_set, val_set, test_set, emb = get_dataset(params,configs)
    else:
        train_set, val_set, test_set = get_dataset(params,configs)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=params["batch_size"], shuffle=True, num_workers=configs['num_workers'])
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=params["batch_size"],shuffle=False, num_workers=configs['num_workers'])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=params["batch_size"],shuffle=False, num_workers=configs['num_workers'])
    # ---------
    # Task loss
    # ---------
    loss_fn = get_loss(params,configs)
    tasks = params["tasks"]
    # ------
    # Solver
    # ------
    solvers = dict(ls=LinearScalarizationSolver, epo=EPOSolver,cheby=ChebyshevBasedSolver, utility = UtilityBasedSolver)
    solver_type = params["solver"]
    if solver_type == "epo":
        solver = EPOSolver(n_tasks=params['num_task'], n_params=count_parameters(hnet))
    elif solver_type == "ls":
        solver = LinearScalarizationSolver()
    elif solver_type == "cheby":
        solver = ChebyshevBasedSolver(lower_bound = 0.1)
    elif solver_type == "utility":
        solver = UtilityBasedSolver(upper_bound = 200.0)
    # ----------
    # Train loop
    # ----------
    best_hv = -1
    start = time.time()
    patience = 0
    early_stop = 0
    for epoch in range(params["epochs"]):
        losses_epoch = []
        losses = np.zeros(params["num_tasks"])
        bs = 0
        if early_stop == params["early_stop"]:
            break
        if (patience+1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= np.sqrt(0.5)
                patience = 0
            lr = param_group['lr']
            print("Reduce Learning rate", param_group['lr'], "at", epoch)
        if configs["name_exp"] == 'mnist':
            print("Train !")
            train_mnist(hnet, net, train_loader, device, params,configs,optimizer,loss_fn,solver)
            print("Evaluate !")
            hv_eval = eval_mnist(hnet, net, val_loader, test_ray, device,params,configs)
        print(" Epoch {}, Best val HV {:.5f}, Early_stop {} ".format(epoch, best_hv, early_stop))
        if hv_eval > best_hv:
            best_hv = hv_eval
            save_dict = {'state_dicts': hnet.state_dict()}
            torch.save(save_dict,"./save_models/best_model_"+str(params["solver"])+"_"+str(model_type)+"_"+params['dataset']+".pkl")
            patience = 0
            early_stop = 0
        else:
            early_stop += 1
            patience += 1 
    end = time.time()
    print("Training time: ",end-start)
    ckpt = torch.load("./save_models/best_model_"+str(params["solver"])+"_"+str(model_type)+"_"+params['dataset']+".pkl")
    hnet.load_state_dict(ckpt['state_dicts'])
    hv_test = eval_mnist(hnet, net, val_loader, test_ray, device,params,configs)
    print("HV on test:",hv_test)

if __name__ == "__main__":
    with open("configs.json") as json_params:
        configs = json.load(json_params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(configs['seed'])
    if configs["name_exp"] == "mnist":
        json_path = "./params/mnist.json"
    elif configs["name_exp"] == "celebA":
        json_path = "./params/celebA.json"
    with open(json_path) as json_params:
        params = json.load(json_params)
    test_ray = get_test_rays(params['num_tasks'],n_partitions = 100)
    rng = np.random.default_rng()
    test_ray = rng.choice(test_ray,params['num_ray_eval'])
    # np.save("eval_rays.npy",test_ray)
    # test_ray = np.load("eval_rays.npy")
    model_type = params['model_type']
    train(params,configs,test_ray,model_type,device)
