import torch
import torch.nn.functional as F
from torch import nn

class HyperCNNMlp(nn.Module):
    def __init__(self, embed_size=300, num_filters=36, ray_hidden_dim=100, dropout=0.1):
        super(HyperCNNMlp, self).__init__()
        self.filter_sizes = [1,2,3,5]
        self.num_filters = num_filters
        self.embed_size = embed_size

        self.n_classes = 100
        self.dropout = nn.Dropout(dropout)

        self.ray_mlp = nn.Sequential(
            nn.Linear(2, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
        )

        for i in range(len(self.filter_sizes)):
            setattr(self, f"conv{i}_weights", nn.Linear(ray_hidden_dim, self.num_filters*self.filter_sizes[i]*self.embed_size))
            setattr(self, f"conv{i}_bias", nn.Linear(ray_hidden_dim, self.num_filters))
            
            setattr(self, f"MSE_weights", nn.Linear(ray_hidden_dim, len(self.filter_sizes)*self.num_filters))
            setattr(self, f"MSE_bias", nn.Linear(ray_hidden_dim, 1))
            setattr(self, f"CE_weights", nn.Linear(ray_hidden_dim, self.n_classes*len(self.filter_sizes)*self.num_filters))
            setattr(self, f"CE_bias", nn.Linear(ray_hidden_dim, self.n_classes))
    def forward(self, ray):
        features = self.ray_mlp(ray)
        out_dict = {}
        for i in range(len(self.filter_sizes)):
            out_dict[f"conv{i}_weights"] = self.dropout(getattr(self, f"conv{i}_weights")(features))
            out_dict[f"conv{i}_bias"] = self.dropout(getattr(self, f"conv{i}_bias")(features).flatten())
            out_dict[f"MSE_weights"] = self.dropout(getattr(self, f"MSE_weights")(features))
            out_dict[f"MSE_bias"] = getattr(self, f"MSE_bias")(features).flatten()
            out_dict[f"CE_weights"] = self.dropout(getattr(self, f"CE_weights")(features))
            out_dict[f"CE_bias"] = getattr(self, f"CE_bias")(features).flatten()
        return out_dict


class HyperCNNTrans(nn.Module):
    def __init__(self, embed_size=300, num_filters=36, ray_hidden_dim=100, dropout=0.1,act_type = 'relu'):
        super(CNN_Hyper_Attention, self).__init__()
        self.filter_sizes = [1,2,3,5]
        self.num_filters = num_filters
        self.embed_size = embed_size
        self.act_type = act_type
        self.n_classes = 100
        self.dropout = nn.Dropout(dropout)

        self.embedding_layer1 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
        self.embedding_layer2 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
        self.attention = nn.MultiheadAttention(embed_dim=ray_hidden_dim, num_heads=1)
        self.ffn1 = nn.Linear(ray_hidden_dim,ray_hidden_dim)
        self.ffn2 = nn.Linear(ray_hidden_dim,ray_hidden_dim)

        self.output_layer = nn.Linear(ray_hidden_dim, ray_hidden_dim)

        for i in range(len(self.filter_sizes)):
            setattr(self, f"conv{i}_weights", nn.Linear(ray_hidden_dim, self.num_filters*self.filter_sizes[i]*self.embed_size))
            setattr(self, f"conv{i}_bias", nn.Linear(ray_hidden_dim, self.num_filters))
            
            setattr(self, f"MSE_weights", nn.Linear(ray_hidden_dim, len(self.filter_sizes)*self.num_filters))
            setattr(self, f"MSE_bias", nn.Linear(ray_hidden_dim, 1))
            setattr(self, f"CE_weights", nn.Linear(ray_hidden_dim, self.n_classes*len(self.filter_sizes)*self.num_filters))
            setattr(self, f"CE_bias", nn.Linear(ray_hidden_dim, self.n_classes))
    def forward(self, ray):
        x = torch.stack((self.embedding_layer1(ray[0].unsqueeze(0)),self.embedding_layer2(ray[1].unsqueeze(0))))
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

        out_dict = {}
        for i in range(len(self.filter_sizes)):
            out_dict[f"conv{i}_weights"] = self.dropout(getattr(self, f"conv{i}_weights")(features))
            out_dict[f"conv{i}_bias"] = self.dropout(getattr(self, f"conv{i}_bias")(features).flatten())
            out_dict[f"MSE_weights"] = self.dropout(getattr(self, f"MSE_weights")(features))
            out_dict[f"MSE_bias"] = getattr(self, f"MSE_bias")(features).flatten()
            out_dict[f"CE_weights"] = self.dropout(getattr(self, f"CE_weights")(features))
            out_dict[f"CE_bias"] = getattr(self, f"CE_bias")(features).flatten()
        return out_dict
  

class CNNTarget(nn.Module):
    
    def __init__(self, embed_size=300, num_filters=36):
        super(CNN_Target, self).__init__()
        self.filter_sizes = [1,2,3,5]
        self.num_filters = num_filters
        self.embed_size = embed_size
        self.n_classes = 100

    def forward(self, x, weights, embedding_matrix):
        x = F.embedding(x, embedding_matrix)  
        x = x.unsqueeze(1)
        x_lst = []
        for i in range(len(self.filter_sizes)):
            x_lst.append(F.relu(F.conv2d(x, weight=weights[f'conv{i}_weights'].reshape(self.num_filters, 1, self.filter_sizes[i], self.embed_size),
                                bias=weights[f'conv{i}_bias'])).squeeze(3))
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x_lst]  
        x = torch.cat(x, 1)
        #x = self.dropout(x)  
        logit_MSE = F.linear(x, weight=weights[f'MSE_weights'].reshape(1, len(self.filter_sizes)*self.num_filters),
                         bias=weights[f'MSE_bias'])
        logit_CE = F.linear(x, weight=weights[f'CE_weights'].reshape(self.n_classes, len(self.filter_sizes)*self.num_filters),
                         bias=weights[f'CE_bias'])
        logits = [logit_MSE, logit_CE]
        return logits
