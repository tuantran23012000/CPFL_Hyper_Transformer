from torch import nn
import torch.nn.functional as F
import torch
class Hypernet_mlp(nn.Module):
    def __init__(self, ray_hidden_dim=30, out_dim=1, target_hidden_dim=15, n_hidden=1, n_tasks=2):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_tasks = n_tasks
        self.out_dim = out_dim
        self.ray_mlp = nn.Sequential(
                nn.Linear(self.n_tasks, ray_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(ray_hidden_dim, ray_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(ray_hidden_dim, ray_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(ray_hidden_dim, ray_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(ray_hidden_dim, ray_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(ray_hidden_dim, ray_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(ray_hidden_dim, ray_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(ray_hidden_dim, out_dim),

            )


    def forward(self, ray):
        x = self.ray_mlp(ray)
        return x

class Hypernet_trans(nn.Module):
    def __init__(self, ray_hidden_dim=30, out_dim=1, target_hidden_dim=15, n_hidden=1, n_tasks=2):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_tasks = n_tasks

        if self.n_tasks == 2:
            self.embedding_layer1 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
            self.embedding_layer2 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
        else:
            self.embedding_layer1 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
            self.embedding_layer2 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
            self.embedding_layer3 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
            
        self.output_layer =  nn.Linear(ray_hidden_dim, out_dim)
        self.attention = nn.MultiheadAttention(embed_dim=ray_hidden_dim, num_heads=2)
        self.ffn1 = nn.Linear(ray_hidden_dim,ray_hidden_dim)
        self.ffn2 = nn.Linear(ray_hidden_dim, ray_hidden_dim)


    def forward(self, ray):
        ray = ray.unsqueeze(0)
        if self.n_tasks == 2: 
            x = torch.stack((self.embedding_layer1(ray[:,0].unsqueeze(1)),self.embedding_layer2(ray[:,1].unsqueeze(1))))
        else:
            x = torch.stack((self.embedding_layer1(ray[:,0].unsqueeze(1)),self.embedding_layer2(ray[:,1].unsqueeze(1)),self.embedding_layer3(ray[:,2].unsqueeze(1))))
        x_ = x
                
        x,_ = self.attention(x,x,x)
        x = x + x_
        x_ = x
        x = self.ffn1(x)
        x = F.relu(x)
        x = self.ffn2(x)
        x = x + x_
        x = self.output_layer(x)
        x = torch.mean(x,dim=0)  
        return x