import numpy as np
import torch
from matplotlib import pyplot as plt
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from pymoo.util.remote import Remote
def get_ref_dirs(n_obj):
    if n_obj == 2:
        ref_dirs = UniformReferenceDirectionFactory(2, n_points=100).do()
    elif n_obj == 3:
        #ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=15).do()
        ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=100).do()
    else:
        raise Exception("Please provide reference directions for more than 3 objectives!")
    return ref_dirs
def generic_sphere(ref_dirs):
    return ref_dirs / np.tile(np.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))
class DTLZ7():
    def __init__(self, n_var=10, n_obj=3):
        #super().__init__(n_var=n_var, n_obj=n_obj)
        self.k = n_var - n_obj + 1
        self.n_obj = n_obj
        self.n_var = n_var
    def create_pf(self):
        if self.n_obj == 3:
            pf = Remote.get_instance().load("pymoo", "pf", "dtlz7-3d.pf")
            pf[:, 2] = pf[:, 2]/6 
            #print(pf)
            return pf
        else:
            raise Exception("Not implemented yet.")
    def f_1(self, output):
        return output[0][0]
    def f_2(self, output):
        return output[0][1]
    def f_3(self, output):
        tmp = 0
        # for i in range(0,self.k):
        #     tmp += torch.sum(output[0,-self.k:])
        g = 1 + (9/(self.k))*(torch.sum(output[0,-self.k:]))
        #print(g)
        
        f1 = output[0][0]
        f2 = output[0][1]
        h = self.n_obj - ((f1/(1+g))*(1+torch.sin(3*torch.pi*f1))) - ((f2/(1+g))*(1+torch.sin(3*torch.pi*f2)))
        return (1+g)*h/6
    def _evaluate(self, x, out,):
        f = []
        for i in range(0, self.n_obj - 1):
            f.append(x[:, i])
        f = anp.column_stack(f)

        g = 1 + 9 / self.k * anp.sum(x[:, -self.k:], axis=1)
        h = self.n_obj - anp.sum(f / (1 + g[:, None]) * (1 + anp.sin(3 * anp.pi * f)), axis=1)

        out["F"] = anp.column_stack([f, (1 + g) * h])
class ZDT3_variant():
    def __init__(self):
        self.n_points = 10000
        self.A = 2
        self.alpha = 3
        self.beta = 1/3
    def create_pf(self):
        # regions = [[0, 0.0830015349],
        #            [0.182228780, 0.2577623634],
        #            [0.4093136748, 0.4538821041],
        #            [0.6183967944, 0.6525117038],
        #            [0.8233317983, 0.8518328654]]
        regions = [[0.01,0.7],
                    #[0.6183967944, 0.6525117038],
                   [0.8, 0.85]]

        pf = []
        flatten = True
        for r in regions:
            x1 = np.linspace(r[0], r[1], int(self.n_points / len(regions)))
            x2 = (1 + (1 - np.sqrt(x1) - (x1**(self.alpha)) * np.sin(self.A * np.pi * (x1**(self.beta)))))/2
            pf.append(np.array([x1, x2]).T)

        if not flatten:
            pf = np.concatenate([pf[None,...] for pf in pf])
        else:
            pf = np.row_stack(pf)

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
        return (1 + g*(1 - torch.sqrt(f1/g) - (f1**(self.alpha)/g)*torch.mean(torch.sin(self.A*torch.pi*(f1**(self.beta))))))/2
class ZDT3():
    def __init__(self):
        self.n_points = 10000
    def create_pf(self):
        regions = [[0, 0.0830015349],
                   [0.182228780, 0.2577623634],
                   [0.4093136748, 0.4538821041],
                   [0.6183967944, 0.6525117038],
                   [0.8233317983, 0.8518328654]]
        
        pf = []
        flatten = True
        for r in regions:
            x1 = np.linspace(r[0], r[1], int(self.n_points / len(regions)))
            x2 = (1 + (1 - np.sqrt(x1) - (x1) * np.sin(10 * np.pi * (x1))))/2
            pf.append(np.array([x1, x2]).T)

        if not flatten:
            pf = np.concatenate([pf[None,...] for pf in pf])
        else:
            pf = np.row_stack(pf)

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
        return (1 + g*(1 - torch.sqrt(f1/g) - (f1/g)*torch.mean(torch.sin(10*torch.pi*(f1)))))/2

class Problem():
    def __init__(self,name, mode):
        self.name = name
        if self.name == 'ZDT3':
            self.pb = ZDT3()
        elif self.name == 'ZDT3_variant':
            self.pb = ZDT3_variant()
        elif self.name == 'DTLZ7':
            self.pb = DTLZ7()
        self.mode = mode
    def get_pf(self):
        pf = self.pb.create_pf()
        return pf
    def get_values(self, output):
        if self.mode == '2d':
            f1, f2 = self.pb.f_1(output), self.pb.f_2(output)
            objectives = [f1, f2]
        else:
            f1, f2, f3 = self.pb.f_1(output), self.pb.f_2(output), self.pb.f_3(output)
            objectives = [f1, f2, f3]
        return objectives

