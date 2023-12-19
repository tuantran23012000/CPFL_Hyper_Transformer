import torch.nn.functional as F
from torch import nn
import torch

class HyperMLPMlp(nn.Module):
    def __init__(self, ray_hidden_dim=50):
        super().__init__()
        self.ray_mlp = nn.Sequential(
            nn.Linear(7, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
        )

        self.in_dim = 21
        # self.dims = [128, 128, 128, 7]
        self.dims = [256, 256, 256, 7]
        prvs_dim = self.in_dim
        for i, dim in enumerate(self.dims):
            setattr(self, f"fc_{i}_weights", nn.Linear(ray_hidden_dim, prvs_dim * dim))
            setattr(self, f"fc_{i}_bias", nn.Linear(ray_hidden_dim, dim))
            prvs_dim = dim

    def forward(self, ray):
        out_dict = dict()
        features = self.ray_mlp(ray)
        prvs_dim = self.in_dim
        for i, dim in enumerate(self.dims):
            out_dict[f"fc_{i}_weights"] = self.__getattr__(f"fc_{i}_weights")(
                features
            ).reshape(dim, prvs_dim)
            out_dict[f"fc_{i}_bias"] = self.__getattr__(f"fc_{i}_bias")(
                features
            ).flatten()
            prvs_dim = dim
        return out_dict

class HyperMLPTrans(nn.Module):
    def __init__(self, ray_hidden_dim=50,act_type='relu'):
        super().__init__()
        self.act_type = act_type
        self.embedding_layer1 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
        self.embedding_layer2 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
        self.embedding_layer3 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
        self.embedding_layer4 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
        self.embedding_layer5 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
        self.embedding_layer6 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
        self.embedding_layer7 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
        self.attention = nn.MultiheadAttention(embed_dim=ray_hidden_dim, num_heads=1)
        self.ffn1 = nn.Linear(ray_hidden_dim,ray_hidden_dim)
        out_hd_dim = ray_hidden_dim
        self.ffn2 = nn.Linear(ray_hidden_dim,ray_hidden_dim)
        self.output_layer = nn.Linear(ray_hidden_dim, out_hd_dim)
        
        self.in_dim = 21
        # self.dims = [128, 128, 128, 7]
        self.dims = [256, 256, 256, 7]
        prvs_dim = self.in_dim
        for i, dim in enumerate(self.dims):
            setattr(self, f"fc_{i}_weights", nn.Linear(ray_hidden_dim, prvs_dim * dim))
            setattr(self, f"fc_{i}_bias", nn.Linear(ray_hidden_dim, dim))
            prvs_dim = dim

    def forward(self, ray):
        out_dict = dict()
        x = torch.stack((self.embedding_layer1(ray[0].unsqueeze(0)),
                         self.embedding_layer2(ray[1].unsqueeze(0)),
                         self.embedding_layer3(ray[2].unsqueeze(0)),
                         self.embedding_layer4(ray[3].unsqueeze(0)),
                         self.embedding_layer5(ray[4].unsqueeze(0)),
                         self.embedding_layer6(ray[5].unsqueeze(0)),
                         self.embedding_layer7(ray[6].unsqueeze(0))))
        x_ = x
        
        x = x.unsqueeze(1)
        x,_ = self.attention(x,x,x)
        x = x.squeeze(1)
        x = x + x_
        x_ = x
        x = self.ffn1(x)
        if self.act_type == 'relu':
            x = F.relu(x)
        else:
            x = F.gelu(x)
        x = self.ffn2(x)
        x = x + x_
        x = self.output_layer(x)
        x = F.relu(x) 
        features = torch.mean(x,dim=0)
        
        prvs_dim = self.in_dim
        for i, dim in enumerate(self.dims):
            out_dict[f"fc_{i}_weights"] = self.__getattr__(f"fc_{i}_weights")(
                features
            ).reshape(dim, prvs_dim)
            out_dict[f"fc_{i}_bias"] = self.__getattr__(f"fc_{i}_bias")(
                features
            ).flatten()
            prvs_dim = dim
        return out_dict


class MLPTarget(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, weights):
        for i in range(int(len(weights) / 2)):
            x = F.linear(x, weights[f"fc_{i}_weights"], weights[f"fc_{i}_bias"])
            if i < int(len(weights) / 2) - 1:
                x = F.relu(x)
        return x