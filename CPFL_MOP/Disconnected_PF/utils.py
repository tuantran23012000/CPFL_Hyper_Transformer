import logging
import random
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.tri import Triangulation, LinearTriInterpolator
from scipy import stats
import itertools
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib
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
def simplex(n_vals):
    base = np.linspace(0, 0.25, n_vals, endpoint=False)
    coords = np.asarray(list(itertools.product(base, repeat=3)))
    return coords[np.isclose(coords.sum(axis=-1), 0.25)]
def set_logger():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
def circle_points_random(r, n):
    """
    generate n random unit vectors
    """
    
    circles = []
    for r, n in zip(r, n):
        t = np.random.rand(n) * 0.5 * np.pi  
        t = np.sort(t)
        
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles

def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def sample_vec(n,m):
    vector = [0]*n
    unit = np.linspace(0, 1, m)
    rays = []
    def sample(i, sum):
        if i == n-1:
            vector[i] = 1-sum
            rays.append(vector.copy())
            return vector
        for value in unit:
            if value > 1-sum:
                break
            else:
                vector[i] = value
                sample(i+1, sum+value)
    sample(0,0)
    return rays

def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = np.pi / 20. if min_angle is None else min_angle
    ang1 = np.pi * 9 / 20. if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]

def find_target(pf, criterion, context,c,cfg):

    if criterion == 'Log':
        F = np.sum(context*np.log(pf+1),axis = 1)

    elif criterion == 'Prod':
        F = np.prod((pf+1)**context,axis = 1)

    elif criterion == 'AC':
        F1 = np.max(context*pf,axis = 1)
        F2 = np.sum(context*pf,axis = 1)
        rho = cfg['TRAIN']['Solver'][criterion]['Rho']
        F = F1 + rho*F2

    elif criterion == 'MC':
        rho = cfg['TRAIN']['Solver'][criterion]['Rho']
        F1 = np.sum(context*pf,axis = 1).reshape(pf.shape[0],1)
        F = np.max(context*pf + rho*F1,axis = 1)

    elif criterion == 'HV':
        n_mo_obj = cfg['TRAIN']['N_task']
        ref_point = tuple(map(int, cfg['TRAIN']['Ref_point'].split(',')))
        rho = cfg['TRAIN']['Solver'][criterion]['Rho'] 
        mo_opt = HvMaximization(1, n_mo_obj, ref_point)
        loss_numpy = pf[:, :,np.newaxis]
        n_samples = loss_numpy.shape[0]
        dynamic_weight = []
        for i_sample in range(0, n_samples):
            dynamic_weight.append((mo_opt.compute_weights(loss_numpy[i_sample,:,:])).reshape(1,n_mo_obj).tolist()[0])
        dynamic_weight = np.array(dynamic_weight)
        rl = np.sum(context*pf,axis = 1)
        l_s = np.sqrt(np.sum(pf**2,axis = 1))
        r_s = np.sqrt(np.sum(np.array(context)**2))
        cosine = - (rl) / (l_s*r_s)
        F = -np.sum((dynamic_weight*pf),axis =1) + rho*cosine
    elif criterion == 'HVI':
        n_mo_obj = cfg['TRAIN']['N_task']
        ref_point = tuple(map(int, cfg['TRAIN']['Ref_point'].split(',')))
        rho = cfg['TRAIN']['Solver'][criterion]['Rho'] 
        mo_opt = HvMaximization(1, n_mo_obj, ref_point)
        loss_numpy = pf[:, :,np.newaxis]
        n_samples = loss_numpy.shape[0]
        dynamic_weight = []
        for i_sample in range(0, n_samples):
            dynamic_weight.append((mo_opt.compute_weights(loss_numpy[i_sample,:,:])).reshape(1,n_mo_obj).tolist()[0])
        dynamic_weight = np.array(dynamic_weight)
        rl = np.sum(context*pf,axis = 1)
        l_s = np.sqrt(np.sum(pf**2,axis = 1))
        r_s = np.sqrt(np.sum(np.array(context)**2))
        cosine = - (rl) / (l_s*r_s)
        # F = -np.sum((dynamic_weight*pf),axis =1) + rho*cosine
        F = -np.sum((dynamic_weight*pf),axis =1) + rho*cosine
    elif criterion == 'Cheby':
        #lower_bound = np.array([0.6,0.6])
        lower_bound = c
        
        F = np.max(context*(pf-lower_bound),axis = 1)

    elif criterion == 'LS':
        F = np.sum(context*pf,axis = 1)

    elif criterion == 'Utility':
        ub = cfg['TRAIN']['Solver'][criterion]['Ub']
        F = 1/np.prod(((ub-pf)**context),axis=1)

    elif criterion == 'Cosine':
        rl = np.sum(context*pf,axis = 1)
        l_s = np.sqrt(np.sum(pf**2,axis = 1))
        r_s = np.sqrt(np.sum(np.array(context)**2))
        F = - (rl) / (l_s*r_s)

    elif criterion == 'KL':
        m = pf.shape[1]
        rl = np.exp(context*pf)
        normalized_rl = rl/np.sum(rl,axis=1).reshape(pf.shape[0],1)
        F = np.sum(normalized_rl * np.log(normalized_rl * m),axis=1) 
    elif criterion == 'EPO':
        m = pf.shape[1]
        rl = context*pf
        normalized_rl = rl/np.sum(rl,axis=1).reshape(pf.shape[0],1)
        F = np.sum(normalized_rl * np.log(normalized_rl * m + 0.01),axis=1) 

    elif criterion == 'Cauchy':
        rl = np.sum(context*pf,axis = 1)
        l_s = np.sum(pf**2,axis = 1)
        r_s = np.sum(np.array(context)**2)
        F = 1 - (rl)**2 / (l_s*r_s)

    return pf[F.argmin(), :]
def get_d_paretomtl(pf,grads,value,normalized_rest_weights,normalized_current_weight):
    w = normalized_rest_weights - normalized_current_weight
        
    #w = normalized_rest_weights
    # solve QP 
    F = []
    for value in pf:
        value = torch.tensor(value).float()
        gx =  torch.matmul(w,value/torch.norm(value))
        idx = gx >  0
        #print(torch.sum(idx))
        if torch.sum(idx) <= 0:
            sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
            #print(sol)
            weight = torch.tensor(sol).float()
            f = (weight*value).sum()
            F.append(f)
            continue
        vec =  torch.cat((grads, torch.matmul(w[idx],grads)))
        # use MinNormSolver to solve QP
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])
        
        # reformulate ParetoMTL as linear scalarization method, return the weights
        weight0 =  sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2,0] for j in torch.arange(2,2 + torch.sum(idx))]))
        weight1 = sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2,1] for j in torch.arange(2,2 + torch.sum(idx))]))
        weight = torch.stack([weight0,weight1])
        f = (weight*value).sum()
        F.append(f)
    F = np.array(F)
    return pf[F.argmin(), :]
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
def visualize_3d(sol,pf,cfg,criterion,pb,model_type):
    x = []
    y = []
    z = []
    for s in sol:
        x.append(s[0])
        y.append(s[1])
        z.append(s[2])
    sim = simplex(5)
    X = sim[:, 0]
    Y = sim[:, 1]
    Z = sim[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    tri = Triangulation(X, Y)
    triangle_vertices = np.array([np.array([[X[T[0]], Y[T[0]], Z[T[0]]],
                                            [X[T[1]], Y[T[1]], Z[T[1]]], 
                                            [X[T[2]], Y[T[2]], Z[T[2]]]]) for T in tri.triangles])
    triangle_vertices = np.append(triangle_vertices,np.array([np.array([[1,0,0],[0.8,0,0.2],[0.8,0.2,0]])/4]),axis=0)
    triangle_vertices = np.append(triangle_vertices,np.array([np.array([[0,1,0],[0,0.8,0.2],[0.2,0.8,0]])/4]),axis=0)
    triangle_vertices = np.append(triangle_vertices,np.array([np.array([[0,0,1],[0.2,0,0.8],[0,0.2,0.8]])/4]),axis=0)
    collection = Poly3DCollection(triangle_vertices,facecolors='blue', edgecolors=None)
    #if criterion == "KL" or criterion == "Cheby"or criterion == "EPO":
    ax.add_collection(collection)
    if pf is not None:
        # ax.plot_trisurf(pf[:, 0], pf[:, 1], pf[:, 2],
        #                 color='r', alpha=0.5, shade=True)
        ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], zdir='z', s=0.01, c='black', depthshade=True)
    
    graph = ax.scatter(np.array(x), np.array(y), np.array(z), zdir='z',marker='.', s=40, c='green', depthshade=False)
    fake2Dline = mpl.lines.Line2D([0], [0], linestyle="none", c='black',
                                marker='s', alpha=0.5)
    fake2Dline1 = mpl.lines.Line2D([0], [0], linestyle="none", c='green',
                                marker='.', alpha=0.5)
    ax.legend([fake2Dline,fake2Dline1], ['Truth Pareto Front','Learned Pareto Front'], numpoints=1)
    
    ax.set_xlabel(r'$f_1$')
    ax.set_ylabel(r'$f_2$')
    ax.set_zlabel(r'$f_3$')
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_zticks([0.2, 0.4, 0.6, 0.8])
    ax.grid(True)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.zaxis.set_rotate_label(True)
    ax.xaxis.set_rotate_label(True)
    [t.set_va('bottom') for t in ax.get_yticklabels()]
    [t.set_ha('center') for t in ax.get_yticklabels()]
    [t.set_va('bottom') for t in ax.get_xticklabels()]
    [t.set_ha('center') for t in ax.get_xticklabels()]
    [t.set_va('center') for t in ax.get_zticklabels()]
    [t.set_ha('center') for t in ax.get_zticklabels()]

    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
    #ax.view_init(elev=15., azim=100.)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    #ax.legend()
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    title = ax.set_title('')
    
    # def update_graph(i):
    #     ax.view_init(5, int(i%360))
    #     plt.pause(.001)
    #     graph._offsets3d = (x[:i+1],y[:i+1],z[:i+1])
    #     title.set_text('Training, iteration: {}'.format(i))
    # ani = FuncAnimation(fig, update_graph,frames = len(x), interval=40)
    ax.view_init(45, 45)
    plt.title('Disconnected Pareto Front',fontsize=15)
    if model_type:
        plt.savefig(str(cfg['NAME'])+"_train_joint.pdf")
    else:
        plt.savefig(str(cfg['NAME'])+"_train.pdf")
    #plt.savefig(str(cfg['NAME'])+"_train.pdf")
    #ani.save('./train_results/train.gif', writer='imagemagick', fps=30)
    print("save done! ")
    plt.show()
def visualize_2d(sol,pf,cfg,criterion,pb,model_type):
    x = []
    y = []
    fig, ax = plt.subplots()
    for s in sol:
        # objectives = pb.get_values(torch.Tensor([s]))
        # x.append(objectives[0].item())
        # y.append(objectives[1].item())
        x.append(s[0])
        y.append(s[1])
    ax.scatter(pf[:,0],pf[:,1],s=20,c='gray',label="Truth Pareto Front ")
    #ax.scatter(x[0],y[0], c = 'k', s = 90,label="Initial Point")
    ax.plot(x[0:2],y[0:2])
    ax.grid(color="k", linestyle="-.", alpha=0.3, zorder=0)
    # for i in range(len(x[3:])):
    #     colors = matplotlib.cm.magma_r(np.linspace(0.1, 0.6, len(x[3:])))
    #     plt.scatter(x[3:],y[3:], color=colors, s = 5,zorder=9)

    plt.scatter(x[3:],y[3:], c = 'green', s = 20,label="Learned Pareto Front")
    # ax.scatter(x[-1],y[-1], c = 'red', s = 20,label="Convergence point")
    plt.title('Disconnected Pareto Front',fontsize=15)
    ax.set_xlabel(r'$f_1$',fontsize=15)
    ax.set_ylabel(r'$f_2$',fontsize=15)
    plt.legend()
    # plt.savefig("./train_results/"+str(cfg['NAME'])+"_train_"+str(criterion)+".png")
    # plt.savefig("./train_results/"+str(cfg['NAME'])+"_train_"+str(criterion)+".pdf")
    if model_type:
        plt.savefig(str(cfg['NAME'])+"_train_joint.pdf")
    else:
        plt.savefig(str(cfg['NAME'])+"_train.pdf")
    plt.show()
def vis_2d(cfg,targets_epo, results1, contexts,pb,pf,criterion,igd,med,c,model_type,join_input):
    #c_s = [[0,0.8],[0.14,0.58],[0.33,0.4],[0.56,0.25],[0.8,0.1]] #ZDT3
    #c_s = [[0,0.8],[0.1,0.6],[0.2,0.4],[0.35,0.22],[0.6,0.1]] #CVX1
    #c_s = [[0,0.6],[0.02,0.4],[0.16,0.2],[0.2,0.15],[0.4,0.02]] #CVX2
    # c_s = [[0,0.8],[0.1,0.6],[0.2,0.4],[0.35,0.22],[0.6,0.1]] #ZDT1
    # c_s = [[0.1,0.9],[0.1,0.6],[0.2,0.4],[0.35,0.22],[0.6,0.1]] #ZDT2
    # c_s = [[0,0.6],[0.79,0.62]] #ZDT3_variant
    name = cfg["NAME"]
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
        # c_s = [[0,0.6],[0.79,0.62]] #ZDT3_variant
        c_s = [[0.79,0.62],[0,0.6]] #ZDT3_variant
    elif name == 'DTLZ2':
        c_s = [[0.15,0.2,0.7],[0.2,0.5,0.6],[0.2,0.7,0.4],[0.35,0.6,0.22],[0.6,0.1,0.46]] #DTLZ2
    elif name == 'DTLZ7':
        c_s = [[0.1,0.7,0.7],[0.1,0.1,0.83],[0.77,0.7,0.5],[0.7,0.1,0.65]] #DTLZ7
    fig, ax = plt.subplots(figsize=(8,6))
    meds = []
    for idx,c in enumerate(c_s):
        if join_input:
            targets_epo = np.load('./predict/target_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c[0])+"_"+str(c[1])+'_join.npy')
            results1 = np.load('./predict/predict_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c[0])+"_"+str(c[1])+'_join.npy')
            med = np.load('./predict/med_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c[0])+"_"+str(c[1])+'_join.npy')
            meds.append(med)
            contexts = np.load('./predict/ray_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c[0])+"_"+str(c[1])+'_join.npy')
        else:
            targets_epo = np.load('./predict/target_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c[0])+"_"+str(c[1])+'.npy')
            results1 = np.load('./predict/predict_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c[0])+"_"+str(c[1])+'.npy')
            med = np.load('./predict/med_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c[0])+"_"+str(c[1])+'.npy')
            meds.append(med)
            contexts = np.load('./predict/ray_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c[0])+"_"+str(c[1])+'.npy')

        mode = cfg['MODE']
        name = cfg['NAME']
        
        colors = ["r","g","y"]
        for k,r in enumerate(contexts):
            r  = r/ r.sum()
            r_inv = 1. / r
            
            #r_inv = r
            if criterion == "KL" or criterion == "Cheby" or criterion == "HVI" or criterion == "EPO":
                root = c
                ep_ray = 0.3 * r_inv / np.linalg.norm(r_inv) + root
                ep_ray1 = 1.1 * r_inv / np.linalg.norm(r_inv)
                # root = np.zeros(2)
                
                ep_ray_line = np.stack([root, ep_ray])
                ep_ray_line1 = np.stack([np.zeros(2), ep_ray1])
                label = r'$mr^{-1}$ ray' if k == 0 else ''
                label1 = r'$r^{-1}$ ray' if k == 0 else ''
                if idx == 0:
                    label = r'$mr^{-1}$ ray' if k == 0 else ''
                else:
                    label = ''
                ax.plot(ep_ray_line[:, 0], ep_ray_line[:, 1], color="g",
                        lw=1, ls='-.', dashes=(15, 5),label=label)
                # ax.plot(ep_ray_line1[:, 0], ep_ray_line1[:, 1], color="y",
                #         lw=1,ls='--', dashes=(15, 5),label=label1)
                # ax.arrow(.95 * ep_ray1[0], .95 * ep_ray1[1],
                #             .05 * ep_ray1[0], .05 * ep_ray1[1],
                #             color="y", lw=1, head_width=.02)
                # ax.arrow(.98 * ep_ray[0], .98 * ep_ray[1],
                #             .02 * ep_ray[0], .02 * ep_ray[1],
                #             color="g", lw=1, head_width=.02)
        if idx == 0:
            label_target = 'Target'
            label_predict = 'Predict'
            label_pf = 'Pareto Front'
        else:
            label_target = ''
            label_predict = ''
            label_pf = ''
        ax.scatter(targets_epo[:,0], targets_epo[:,1], s=40,c='red', marker='D', alpha=1,label=label_target)
        ax.scatter(results1[:, 0], results1[:, 1],s=40,c='green',marker='*',label=label_predict) #HPN-PNGD
        tmp0 = []
        tmp1 = []
        for i in range(pf.shape[0]):
            if pf[i][0] >= c[0] and pf[i][1] >= c[1]:
                tmp0.append(pf[i,0])
                tmp1.append(pf[i,1])
        ax.scatter(pf[:,0],pf[:,1],lw=1,c='gray',label=label_pf,zorder=0)
        #ax.scatter(tmp0,tmp1,lw=5,c='gray',label='A part of PF',zorder=0)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        #ax.grid(color="k", linestyle="-.", alpha=0.3, zorder=0)
        ax.set_xlabel(r'$f_1$',fontsize=30)
        ax.set_ylabel(r'$f_2$',fontsize=30)
        # c_s = [[0.1,0.8],[0.14,0.58],[0.33,0.4],[0.56,0.25],[0.8,0.1]]
        #for i,c_ in enumerate(c_s):
        if idx == 0:
            label = 'Lower bound'
        else:
            label = ''
        ax.plot([c[0],c[0]],[c[1],c[1]+0.1],ls='-.',c='k',label=label)
        ax.plot([c[0],c[0]+0.33],[c[1],c[1]],ls='-.',c='k')
        ax.legend(fontsize=15)
    plt.title("MED: {0:4f} ".format(np.mean(np.array(meds))),fontsize=30)
    #ax.set_title("MED :{0:4f}".format(med),fontsize=20)
    plt.tight_layout()
    # plt.savefig("./infer_results/"+str(name)+"_"+str(criterion)+"_"+str(mode)+".png")
    # if join_input:
    #     plt.savefig("./out/"+str(name)+"_hyper_"+model_type+"_w_joint.pdf")
    # else:
    #     plt.savefig("./out/"+str(name)+"_hyper_"+model_type+"_wo_joint.pdf")
    
    plt.show()
def visualize_predict_2d(cfg,targets_epo, results1, contexts,pb,pf,criterion,igd,med,c,model_type,join_input):
    mode = cfg['MODE']
    name = cfg['NAME']
    fig, ax = plt.subplots(figsize=(8,6))
    colors = ["r","g","y"]
    for k,r in enumerate(contexts):
        r  = r/ r.sum()
        r_inv = 1. / r
        
        #r_inv = r
        if criterion == "KL" or criterion == "Cheby" or criterion == "HVI" or criterion == "EPO":
            root = c
            ep_ray = 0.7 * r_inv / np.linalg.norm(r_inv) + root
            ep_ray1 = 1.1 * r_inv / np.linalg.norm(r_inv)
            # root = np.zeros(2)
            
            ep_ray_line = np.stack([root, ep_ray])
            ep_ray_line1 = np.stack([np.zeros(2), ep_ray1])
            label = r'$mr^{-1}$ ray' if k == 0 else ''
            label1 = r'$r^{-1}$ ray' if k == 0 else ''
            ax.plot(ep_ray_line[:, 0], ep_ray_line[:, 1], color="g",
                    lw=1, ls='-.', dashes=(15, 5),label=label)
            ax.plot(ep_ray_line1[:, 0], ep_ray_line1[:, 1], color="y",
                    lw=1,ls='--', dashes=(15, 5),label=label1)
            ax.arrow(.95 * ep_ray1[0], .95 * ep_ray1[1],
                        .05 * ep_ray1[0], .05 * ep_ray1[1],
                        color="y", lw=1, head_width=.02)
            ax.arrow(.98 * ep_ray[0], .98 * ep_ray[1],
                        .02 * ep_ray[0], .02 * ep_ray[1],
                        color="g", lw=1, head_width=.02)
    ax.scatter(targets_epo[:,0], targets_epo[:,1], s=40,c='red', marker='D', alpha=1,label='Target')
    ax.scatter(results1[:, 0], results1[:, 1],s=40,c='green',marker='*',label='Predict') #HPN-PNGD
    tmp0 = []
    tmp1 = []
    for i in range(pf.shape[0]):
        if pf[i][0] >= c[0] and pf[i][1] >= c[1]:
            tmp0.append(pf[i,0])
            tmp1.append(pf[i,1])
    ax.scatter(pf[:,0],pf[:,1],lw=5,c='black',label='PF',zorder=0)
    ax.scatter(tmp0,tmp1,lw=5,c='gray',label='A part of PF',zorder=0)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.grid(color="k", linestyle="-.", alpha=0.3, zorder=0)
    ax.set_xlabel(r'$f_1$',fontsize=30)
    ax.set_ylabel(r'$f_2$',fontsize=30)
    
    ax.plot([c[0],c[0]],[0,1],ls='-.',c='k',label='Lower bound')
    ax.plot([0,1],[c[1],c[1]],ls='-.',c='k')
    ax.legend(fontsize=15)
    plt.title("MED: "+str(med),fontsize=30)
    plt.tight_layout()
    # plt.savefig("./infer_results/"+str(name)+"_"+str(criterion)+"_"+str(mode)+".png")
    # if join_input:
    #     plt.savefig("./out/"+str(name)+"_hyper_"+model_type+"_w_joint.pdf")
    # else:
    #     plt.savefig("./out/"+str(name)+"_hyper_"+model_type+"_wo_joint.pdf")

    plt.show()

def concat_2d(cfg,targets_epo, results1, contexts,pb,pf,criterion,igd,med,c):
    model_types = ['mlp','trans']
    join_inputs = [False,True]
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.subplots_adjust(hspace=0.5)
    count = -1
    for join_input in join_inputs:
        for model_type in model_types:
            count += 1
            ax = axes[1, count % 4]
            # X = population_data[[country]].values
            # X = (np.array(X)/1e9)    
            # y = gdp_data[[country]].values
            # y = (np.array(y)/ 1e12)
            # model = LinearRegression()
            # model.fit(X, y)
            # y_pred = model.predict(X)
            # ax.scatter(X, y, label=country, alpha=0.7)
            # ax.plot(X, y_pred, color='red', linewidth=2, label='Linear Regression')
            # ax.set_xlabel('Population')
            # ax.set_ylabel('GDP')
            # ax.set_title(country)
            # ax.legend()
            mode = cfg['MODE']
            name = cfg['NAME']
            #fig, ax = plt.subplots(figsize=(8,6))
            colors = ["r","g","y"]
            for k,r in enumerate(contexts):
                r  = r/ r.sum()
                r_inv = 1. / r
                
                #r_inv = r
                if criterion == "KL" or criterion == "Cheby" or criterion == "HVI" or criterion == "EPO":
                    root = np.array([0.2,0.2])
                    ep_ray = 0.7 * r_inv / np.linalg.norm(r_inv) + root
                    ep_ray1 = 1.1 * r_inv / np.linalg.norm(r_inv)
                    # root = np.zeros(2)
                    
                    ep_ray_line = np.stack([root, ep_ray])
                    ep_ray_line1 = np.stack([np.zeros(2), ep_ray1])
                    label = r'$mr^{-1}$ ray' if k == 0 else ''
                    label1 = r'$r^{-1}$ ray' if k == 0 else ''
                    ax.plot(ep_ray_line[:, 0], ep_ray_line[:, 1], color="g",
                            lw=1, ls='-.', dashes=(15, 5),label=label)
                    ax.plot(ep_ray_line1[:, 0], ep_ray_line1[:, 1], color="y",
                            lw=1,ls='--', dashes=(15, 5),label=label1)
                    ax.arrow(.95 * ep_ray1[0], .95 * ep_ray1[1],
                                .05 * ep_ray1[0], .05 * ep_ray1[1],
                                color="y", lw=1, head_width=.02)
                    ax.arrow(.98 * ep_ray[0], .98 * ep_ray[1],
                                .02 * ep_ray[0], .02 * ep_ray[1],
                                color="g", lw=1, head_width=.02)
            ax.scatter(targets_epo[:,0], targets_epo[:,1], s=40,c='red', marker='D', alpha=1,label='Target')
            ax.scatter(results1[:, 0], results1[:, 1],s=40,c='black',marker='o',label='Predict') #HPN-PNGD
            tmp0 = []
            tmp1 = []
            for i in range(pf.shape[0]):
                if pf[i][0] >= 0.2 and pf[i][1] >= 0.2:
                    tmp0.append(pf[i,0])
                    tmp1.append(pf[i,1])
            ax.scatter(pf[:,0],pf[:,1],lw=5,c='black',label='PF',zorder=0)
            ax.scatter(tmp0,tmp1,lw=5,c='gray',label='A part of PF',zorder=0)
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.grid(color="k", linestyle="-.", alpha=0.3, zorder=0)
            ax.set_xlabel(r'$f_1$',fontsize=30)
            ax.set_ylabel(r'$f_2$',fontsize=30)
            
            ax.plot([c[0],c[0]],[0,1],ls='-.',c='k',label='Lower bound')
            ax.plot([0,1],[c[1],c[1]],ls='-.',c='k')
            ax.legend(fontsize=8)
            ax.set_title("MED :{0:4f}".format(med),fontsize=20)
            #ax.tight_layout()
            # plt.savefig("./infer_results/"+str(name)+"_"+str(criterion)+"_"+str(mode)+".png")
        if join_input:
            plt.savefig("./out/"+str(name)+"_hyper_"+model_type+"_w_joint.pdf")
        else:
            plt.savefig("./out/"+str(name)+"_hyper_"+model_type+"_wo_joint.pdf")
    plt.savefig("./out/"+str(name)+".pdf")
    plt.tight_layout()
    plt.show()

def visualize_predict_3d(cfg,targets_epo, results1, contexts,pb,pf,criterion,igd,med,model_type,join_input):
    mode = cfg['MODE']
    name = cfg['NAME']
    sim = simplex(5)
    x = sim[:, 0]
    y = sim[:, 1]
    z = sim[:, 2]
    tri = Triangulation(x, y)
    triangle_vertices = np.array([np.array([[x[T[0]], y[T[0]], z[T[0]]],
                                            [x[T[1]], y[T[1]], z[T[1]]], 
                                            [x[T[2]], y[T[2]], z[T[2]]]]) for T in tri.triangles])
    triangle_vertices = np.append(triangle_vertices,np.array([np.array([[1,0,0],[0.8,0,0.2],[0.8,0.2,0]])/4]),axis=0)
    triangle_vertices = np.append(triangle_vertices,np.array([np.array([[0,1,0],[0,0.8,0.2],[0.2,0.8,0]])/4]),axis=0)
    triangle_vertices = np.append(triangle_vertices,np.array([np.array([[0,0,1],[0.2,0,0.8],[0,0.2,0.8]])/4]),axis=0)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax = plt.axes(projection ='3d') 
    collection = Poly3DCollection(triangle_vertices,facecolors='blue', edgecolors=None)
    if criterion == "KL" or criterion == "Cheby"or criterion == "EPO":
        ax.add_collection(collection)
    k = 0
    ep_ray_lines = []
    for r in contexts:
        #print(r)
        r_inv = 1. / r
        
        #r_inv  = r_inv/ r_inv.sum()
        #print(r_inv)
        if criterion == "KL" or criterion == "EPO" or criterion == "Cheby":
            root = np.array([0.2,0.2,0.2])
            ep_ray = 0.7 * r_inv / np.linalg.norm(r_inv) + root
            ep_ray1 = 1.0 * r_inv / np.linalg.norm(r_inv)
            #ep_ray  = 2*ep_ray/ ep_ray.sum()
            #print(ep_ray)
            ep_ray_line = np.stack([root, ep_ray])
            ep_ray_line1 = np.stack([np.zeros(3), ep_ray1])
            label = r'$mr^{-1}$ ray' if k == 0 else ''
            label1 = r'$r^{-1}$ ray' if k == 0 else ''
            k+=1
            ep_ray_lines.append(ep_ray_line)
            # ax.plot(ep_ray_line[:, 0], ep_ray_line[:, 1],ep_ray_line[:, 2], color='y',
            #         lw=1, ls='--',label=label)
            # ax.plot(ep_ray_line1[:, 0], ep_ray_line1[:, 1],ep_ray_line1[:, 2], color='g',
            #         lw=1, ls='--',label=label1)
            #print([ep_ray_line[0,0],ep_ray_line[1,0]])
            k = 1.0
            arw = Arrow3D([ep_ray_line[0,0],k*ep_ray_line[1,0]],[ep_ray_line[0,1],k*ep_ray_line[1,1]],[ep_ray_line[0,2],k*ep_ray_line[1,2]], arrowstyle="->", color="g", lw = 1, mutation_scale=15,label=label)
            ax.add_artist(arw)
            arw1 = Arrow3D([ep_ray_line1[0,0],k*ep_ray_line1[1,0]],[ep_ray_line1[0,1],k*ep_ray_line1[1,1]],[ep_ray_line1[0,2],k*ep_ray_line1[1,2]], arrowstyle="->", color="y", lw = 1, mutation_scale=15,label=label1)
            ax.add_artist(arw1)
            # ax.arrow3D(.95 * ep_ray_line[0], .95 * ep_ray_line[1],.95 * ep_ray_line[2],
            #             .05 * ep_ray_line[0], .05 * ep_ray_line[1],.05 * ep_ray_line[2],
            #             color='k', lw=1, head_width=.02)
    x = results1[:,0]
    y = results1[:,1]
    z = results1[:,2]
    x_target = targets_epo[:,0]
    y_target = targets_epo[:,1]
    z_target = targets_epo[:,2]
    #ax.plot_trisurf(pf[:, 0], pf[:, 1], pf[:, 2],color='grey',alpha=0.5, shade=True,antialiased = True)
    tmp0 = []
    tmp1 = []
    tmp2 = []
    tmp0_ = []
    tmp1_ = []
    tmp2_ = []
    for i in range(pf.shape[0]):
        if pf[i][0] >= 0.2 and pf[i][1] >= 0.2 and pf[i][2] >= 0.2:
            tmp0.append(pf[i,0])
            tmp1.append(pf[i,1])
            tmp2.append(pf[i,2])
        else:
            tmp0_.append(pf[i,0])
            tmp1_.append(pf[i,1])
            tmp2_.append(pf[i,2])
    #ax.plot_trisurf(pf[:, 0], pf[:, 1], pf[:, 2],color='black',alpha=0.5, shade=True,antialiased = True)
    # ax.plot_trisurf(tmp0,tmp1,tmp2,color='grey',alpha=0.5, shade=True,antialiased = True)
    ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], zdir='z', s=10, c='grey', depthshade=True)
    #ax.scatter(tmp0_,tmp1_,tmp2_, zdir='z', s=10, c='black', depthshade=True)
    # ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], zdir='z', s=10, c='black', depthshade=True,label = 'Predict')
    #ax.scatter(tmp0,tmp1,tmp2, zdir='z', s=20, c='grey', depthshade=True,label = 'Target')


    ax.scatter(x, y, z, zdir='z',marker='*', s=40, c='green', depthshade=False,label = 'Predict')
    ax.scatter(x_target, y_target, z_target, zdir='z',marker='D', s=40, c='red', depthshade=False,label = 'Target')
    ax.set_xlabel(r'$f_1$',fontsize=18)
    ax.set_ylabel(r'$f_2$',fontsize=18)
    ax.set_zlabel(r'$f_3$',fontsize=18)
    ax.grid(True)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_zticks([0.2, 0.4, 0.6, 0.8])
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.zaxis.set_rotate_label(False)
    ax.xaxis.set_rotate_label(False)
    [t.set_va('bottom') for t in ax.get_yticklabels()]
    [t.set_ha('center') for t in ax.get_yticklabels()]
    [t.set_va('bottom') for t in ax.get_xticklabels()]
    [t.set_ha('center') for t in ax.get_xticklabels()]
    [t.set_va('center') for t in ax.get_zticklabels()]
    [t.set_ha('center') for t in ax.get_zticklabels()]
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
    #ax.view_init(40, 45) # CVX3
    ax.view_init(40, 45)
    graph = ax.scatter(np.array(x), np.array(y), np.array(z), zdir='z',marker='.', s=10, c='black', depthshade=False)
    title = ax.set_title('')
    fake2Dline = mpl.lines.Line2D([0], [0], linestyle="none", c='grey',
                                marker='s', alpha=0.5)
    fake2Dline1 = mpl.lines.Line2D([0], [0], linestyle="none", c='black',
                                marker='s', alpha=0.5)
    
    # plt.title("MED: "+str(med),fontsize=20)
    # def update_graph(i):
    #     ax.view_init(5, int(i%360))  
    #     graph._offsets3d = (x[:i+1],y[:i+1],z[:i+1])
    #     plt.pause(.001)
    #     title.set_text('Testing, MED: {:.2f}'.format(med))
    # ani = FuncAnimation(fig, update_graph,frames = 360, interval=40)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    X = np.linspace(0.2,1,2)
    Y = np.linspace(0.2,1,2)
    X, Y = np.meshgrid(X, Y)
    Z = 0.2*np.ones((2,2))
    ax.plot_wireframe(X, Y, Z,color='k',label = 'Lower bound')
    X = np.linspace(0.2,1,2)
    Z = np.linspace(0.2,1,2)
    X, Z = np.meshgrid(X, Z)
    Y = 0.2*np.ones((2,2))
    ax.plot_wireframe(X, Y, Z,color='k')
    Y = np.linspace(0.2,1,2)
    Z = np.linspace(0.2,1,2)
    Y, Z = np.meshgrid(Y, Z)
    X = 0.2*np.ones((2,2))
    ax.plot_wireframe(X, Y, Z,color='k')
    #ax.legend(fontsize=8)
    ax.set_title("MED :{0:4f}".format(med),fontsize=15)
    # if count == 0:
    #     ax.set_zlabel("DTLZ2", fontsize=10)
    #plt.tight_layout()
    #ax.legend([fake2Dline,fake2Dline1], ['PF','A part of PF'], numpoints=1,fontsize=15)
    #ani.save('./train_results/test3.gif', writer='imagemagick', fps=30)
    # if join_input:
    #     plt.savefig("./out/"+str(name)+"_hyper_"+model_type+"_w_joint.pdf")
    # else:
    #     plt.savefig("./out/"+str(name)+"_hyper_"+model_type+"_wo_joint.pdf")
    #plt.savefig("./out/"+str(name)+"_hyper_mlp_w_joint.pdf")
    plt.show()
def vis_3d(cfg,targets_epo, results1, contexts,pb,pf,criterion,igd,med,model_type,join_input):
    mode = cfg['MODE']
    name = cfg['NAME']
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
    elif name == 'ZDT3_variant':
        c_s = [[0,0.6],[0.79,0.62]] #ZDT3_variant
    elif name == 'DTLZ2':
        c_s = [[0.15,0.2,0.7],[0.2,0.5,0.6],[0.2,0.7,0.4],[0.35,0.6,0.22],[0.6,0.1,0.46]] #DTLZ2
    elif name == 'DTLZ7':
        c_s = [[0.6,0.6,0.4],[0.01,0.5,0.5],[0.01,0.01,0.7],[0.5,0.01,0.6]]
    sim = simplex(5)
    x = sim[:, 0]
    y = sim[:, 1]
    z = sim[:, 2]
    tri = Triangulation(x, y)
    triangle_vertices = np.array([np.array([[x[T[0]], y[T[0]], z[T[0]]],
                                            [x[T[1]], y[T[1]], z[T[1]]], 
                                            [x[T[2]], y[T[2]], z[T[2]]]]) for T in tri.triangles])
    triangle_vertices = np.append(triangle_vertices,np.array([np.array([[1,0,0],[0.8,0,0.2],[0.8,0.2,0]])/4]),axis=0)
    triangle_vertices = np.append(triangle_vertices,np.array([np.array([[0,1,0],[0,0.8,0.2],[0.2,0.8,0]])/4]),axis=0)
    triangle_vertices = np.append(triangle_vertices,np.array([np.array([[0,0,1],[0.2,0,0.8],[0,0.2,0.8]])/4]),axis=0)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax = plt.axes(projection ='3d') 
    collection = Poly3DCollection(triangle_vertices,facecolors='blue', edgecolors=None)
    if criterion == "KL" or criterion == "Cheby"or criterion == "EPO":
        ax.add_collection(collection)
    # c_s = [[0.15,0.2,0.7],[0.2,0.5,0.6],[0.2,0.7,0.4],[0.35,0.6,0.22],[0.6,0.1,0.46]] #CVX3
    # c_s = [[0.1,0.7,0.7],[0.1,0.1,0.83],[0.77,0.7,0.5],[0.7,0.1,0.65]] #DTLZ7
    #c_s = [[0.15,0.2,0.7],[0.2,0.5,0.6],[0.2,0.7,0.4],[0.35,0.6,0.22],[0.6,0.1,0.46]] #DTLZ2
    meds = []
    for idx,c in enumerate(c_s):
        if join_input:
            targets_epo = np.load('./predict/target_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c[0])+"_"+str(c[1])+'_join.npy')
            results1 = np.load('./predict/predict_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c[0])+"_"+str(c[1])+'_join.npy')
            med = np.load('./predict/med_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c[0])+"_"+str(c[1])+'_join.npy')
            meds.append(med)
            contexts = np.load('./predict/ray_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c[0])+"_"+str(c[1])+'_join.npy')
        else:
            targets_epo = np.load('./predict/target_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c[0])+"_"+str(c[1])+'.npy')
            results1 = np.load('./predict/predict_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c[0])+"_"+str(c[1])+'.npy')
            med = np.load('./predict/med_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c[0])+"_"+str(c[1])+'.npy')
            meds.append(med)
            contexts = np.load('./predict/ray_'+str(cfg["NAME"])+"_"+ str(model_type)+"_"+str(c[0])+"_"+str(c[1])+'.npy')
        k = 0
        ep_ray_lines = []
        for r in contexts:
            #print(r)
            r_inv = 1. / r
            
            #r_inv  = r_inv/ r_inv.sum()
            #print(r_inv)
            if criterion == "KL" or criterion == "EPO" or criterion == "Cheby":
                root = np.array(c)
                ep_ray = 0.5 * r_inv / np.linalg.norm(r_inv) + root
                ep_ray1 = 1.0 * r_inv / np.linalg.norm(r_inv)
                #ep_ray  = 2*ep_ray/ ep_ray.sum()
                #print(ep_ray)
                ep_ray_line = np.stack([root, ep_ray])
                ep_ray_line1 = np.stack([np.zeros(3), ep_ray1])
                if idx == 0 and k==0:
                    label = r'$mr^{-1}$ ray'
                else:
                    label = ''
                #label1 = r'$r^{-1}$ ray' if k == 0 else ''
                k+=1
                ep_ray_lines.append(ep_ray_line)
                # ax.plot(ep_ray_line[:, 0], ep_ray_line[:, 1],ep_ray_line[:, 2], color='y',
                #         lw=1, ls='--',label=label)
                # ax.plot(ep_ray_line1[:, 0], ep_ray_line1[:, 1],ep_ray_line1[:, 2], color='g',
                #         lw=1, ls='--',label=label1)
                #print([ep_ray_line[0,0],ep_ray_line[1,0]])
                k = 1.0
                arw = Arrow3D([ep_ray_line[0,0],k*ep_ray_line[1,0]],[ep_ray_line[0,1],k*ep_ray_line[1,1]],[ep_ray_line[0,2],k*ep_ray_line[1,2]], arrowstyle="->", color="g", lw = 1, mutation_scale=15,label=label)
                ax.add_artist(arw)
                # arw1 = Arrow3D([ep_ray_line1[0,0],k*ep_ray_line1[1,0]],[ep_ray_line1[0,1],k*ep_ray_line1[1,1]],[ep_ray_line1[0,2],k*ep_ray_line1[1,2]], arrowstyle="->", color="y", lw = 1, mutation_scale=15,label=label1)
                # ax.add_artist(arw1)
                # ax.arrow3D(.95 * ep_ray_line[0], .95 * ep_ray_line[1],.95 * ep_ray_line[2],
                #             .05 * ep_ray_line[0], .05 * ep_ray_line[1],.05 * ep_ray_line[2],
                #             color='k', lw=1, head_width=.02)
        x = results1[:,0]
        y = results1[:,1]
        z = results1[:,2]
        x_target = targets_epo[:,0]
        y_target = targets_epo[:,1]
        z_target = targets_epo[:,2]
        #ax.plot_trisurf(pf[:, 0], pf[:, 1], pf[:, 2],color='grey',alpha=0.5, shade=True,antialiased = True)
        tmp0 = []
        tmp1 = []
        tmp2 = []
        tmp0_ = []
        tmp1_ = []
        tmp2_ = []
        for i in range(pf.shape[0]):
            if pf[i][0] >= c[0] and pf[i][1] >= c[1] and pf[i][2] >= c[2]:
                tmp0.append(pf[i,0])
                tmp1.append(pf[i,1])
                tmp2.append(pf[i,2])
            else:
                tmp0_.append(pf[i,0])
                tmp1_.append(pf[i,1])
                tmp2_.append(pf[i,2])
        #ax.plot_trisurf(pf[:, 0], pf[:, 1], pf[:, 2],color='black',alpha=0.5, shade=True,antialiased = True)
        #ax.plot_trisurf(pf[:, 0], pf[:, 1], pf[:, 2],color='grey',alpha=0.5, shade=True,antialiased = True)
        if idx == 0:
            #ax.plot_trisurf(pf[:, 0], pf[:, 1], pf[:, 2],color='grey',alpha=0.5, shade=True,antialiased = True)
            ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], zdir='z', s=0.1, c='grey', depthshade=True)
        # ax.plot_trisurf(tmp0,tmp1,tmp2,color='grey',alpha=0.5, shade=True,antialiased = True)
        # ax.scatter(tmp0_,tmp1_,tmp2_, zdir='z', s=10, c='black', depthshade=True)
        

        if idx == 0:
            label_target = 'Target'
            label_predict = 'Predict'
            label_pf = 'Pareto Front'
            label_l = 'Lower bound'
        else:
            label_target = ''
            label_predict = ''
            label_pf = ''
            label_l = ''
        ax.scatter(x, y, z, zdir='z',marker='*', s=40, c='green', depthshade=False,label = label_predict)
        ax.scatter(x_target, y_target, z_target, zdir='z',marker='D', s=10, c='red', depthshade=False,label = label_target)
        ax.set_xlabel(r'$f_1$',fontsize=18)
        ax.set_ylabel(r'$f_2$',fontsize=18)
        ax.set_zlabel(r'$f_3$',fontsize=18)
        ax.grid(True)
        ax.set_xticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_zticks([0.2, 0.4, 0.6, 0.8])
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_edgecolor('black')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.zaxis.set_rotate_label(False)
        ax.xaxis.set_rotate_label(False)
        [t.set_va('bottom') for t in ax.get_yticklabels()]
        [t.set_ha('center') for t in ax.get_yticklabels()]
        [t.set_va('bottom') for t in ax.get_xticklabels()]
        [t.set_ha('center') for t in ax.get_xticklabels()]
        [t.set_va('center') for t in ax.get_zticklabels()]
        [t.set_ha('center') for t in ax.get_zticklabels()]
        ax.xaxis._axinfo['tick']['inward_factor'] = 0
        ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
        ax.yaxis._axinfo['tick']['inward_factor'] = 0
        ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
        ax.zaxis._axinfo['tick']['inward_factor'] = 0
        ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
        #ax.view_init(40, 45) # CVX3
        ax.view_init(45, 45)
        # graph = ax.scatter(np.array(x), np.array(y), np.array(z), zdir='z',marker='.', s=10, c='black', depthshade=False)
        # title = ax.set_title('')
        fake2Dline = mpl.lines.Line2D([0], [0], linestyle="none", c='grey',
                                    marker='s', alpha=0.5)
        # fake2Dline1 = mpl.lines.Line2D([0], [0], linestyle="none", c='black',
        #                             marker='s', alpha=0.5)
        
        # plt.title("MED: "+str(med),fontsize=20)
        # def update_graph(i):
        #     ax.view_init(5, int(i%360))  
        #     graph._offsets3d = (x[:i+1],y[:i+1],z[:i+1])
        #     plt.pause(.001)
        #     title.set_text('Testing, MED: {:.2f}'.format(med))
        # ani = FuncAnimation(fig, update_graph,frames = 360, interval=40)

        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        # ax.set_zlim(0, 1)

        # X = np.linspace(c[0],1,2)
        # Y = np.linspace(c[1],1,2)
        # X, Y = np.meshgrid(X, Y)
        # Z = c[2]*np.ones((2,2))
        # ax.plot_wireframe(X, Y, Z,lw = 1,color='k',label = label_l)
        # X = np.linspace(c[0],1,2)
        # Z = np.linspace(c[2],1,2)
        # X, Z = np.meshgrid(X, Z)
        # Y = c[1]*np.ones((2,2))
        # ax.plot_wireframe(X, Y, Z,lw = 1,color='k')
        # Y = np.linspace(c[1],1,2)
        # Z = np.linspace(c[2],1,2)
        # Y, Z = np.meshgrid(Y, Z)
        # X = c[0]*np.ones((2,2))
        # ax.plot_wireframe(X, Y, Z,lw = 1,color='k')
        #ax.legend(fontsize=8)
        #ax.set_title("MED :{0:4f}".format(med),fontsize=15)
        # if count == 0:
        #     ax.set_zlabel("DTLZ2", fontsize=10)
        #plt.tight_layout()
        ax.legend([fake2Dline], [label_pf], numpoints=1,fontsize=15)
        ax.legend(fontsize=8)
        #ani.save('./train_results/test3.gif', writer='imagemagick', fps=30)
    plt.title("MED: {0:4f} ".format(np.mean(np.array(meds))),fontsize=30)
    plt.tight_layout()
    # if join_input:
    #     plt.savefig("./out/"+str(name)+"_hyper_"+model_type+"_w_joint.pdf")
    # else:
    #     plt.savefig("./out/"+str(name)+"_hyper_"+model_type+"_wo_joint.pdf")
    plt.savefig("test.png")
    plt.show()