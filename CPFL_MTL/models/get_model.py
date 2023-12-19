import os
print(os.getcwd())
from models.cnn import HyperCNNMlp, HyperCNNTrans, CNNTarget
from models.lenet import HyperLenetMlp, HyperLenetTrans, LenetTarget
from models.mlp import HyperMLPMlp, HyperMLPTrans, MLPTarget
from models.resnet import HyperResnetMlp, HyperResnetTrans, ResnetTarget
from models.segnet import HyperSegnetMlp, HyperSegnetTrans, SegnetTarget

def get_model(params,configs):
    if configs["name_exp"] == "celebA":
        hn_config = {
            "resnet18": {"num_chunks": 105, "num_ws": 11,"model_name": "resnet18"},
            "resnet34": {"num_chunks": 105, "num_ws": 21,"model_name": "resnet34"},
            "resnet50": {"num_chunks": 105, "num_ws": 41,"model_name": "resnet50"},
            "resnet101": {"num_chunks": 105, "num_ws": 61,"model_name": "resnet101"},
        }
        if params["model_type"] == 'mlp':
            hnet = HyperResnetMlp(hidden_dim=params["hidden_dim"], **hn_config[params["backbone"]],out_dim = 2)
        elif params["model_type"] == 'trans_relu':
            hnet = HyperResnetTrans(hidden_dim=params["hidden_dim"], **hn_config[params["backbone"]],out_dim = 2,act_type = 'relu')
        else:
            hnet = HyperResnetTrans(hidden_dim=params["hidden_dim"], **hn_config[params["backbone"]],out_dim = 2,act_type = 'gelu')
        net = ResnetTarget(model_name = params["backbone"],n_tasks = params["num_tasks"])
        return hnet, net
    elif configs["name_exp"] == "mnist":
        if params["model_type"] == 'mlp':
            hnet = HyperLenetMlp([9, 5], ray_hidden_dim=params['hidden_dim'])
        elif params["model_type"] == 'trans_relu':
            hnet = HyperLenetTrans([9, 5], ray_hidden_dim=params['hidden_dim'],act_type = 'relu')
        else:
            hnet = HyperLenetTrans([9, 5], ray_hidden_dim=params['hidden_dim'],act_type = 'gelu')
        net = LenetTarget([9, 5])
        return hnet, net
    elif configs["name_exp"] == "nlp":
        if params["model_type"] == 'mlp':
            hnet = HyperCNNMlp(params["embed_size"], params["num_filters"], params["hidden_dim"], dropout=params["dropout"], act_type = params["model_type"])
        elif params["model_type"] == 'trans_relu':
            hnet = HyperCNNTrans(params["embed_size"], params["num_filters"], params["hidden_dim"], dropout=params["dropout"], act_type = 'relu')
        else:
            hnet = HyperLenetTrans(params["embed_size"], params["num_filters"], params["hidden_dim"], dropout=params["dropout"], act_type = 'gelu')
        net = CNNTarget(params["embed_size"], params["num_filters"])
        
        return hnet, net
    elif configs["name_exp"] == "nuyv2":
        hn_config = {
            "11M": {"num_chunks": 105, "num_ws": 31},
        }
        if params["model_type"] == 'mlp':
            hnet = HyperSegnetMlp(hidden_dim=params["hidden_dim"], **hn_config["11M"])
        elif params["model_type"] == 'trans_relu':
            hnet = HyperSegnetTrans(hidden_dim=params["hidden_dim"], **hn_config["11M"],act_type='relu')
        else:
            hnet = HyperSegnetTrans(hidden_dim=params["hidden_dim"], **hn_config["11M"],act_type='gelu')
        net = SegnetTarget(n_tasks = params["num_tasks"])
        return hnet, net
    elif configs["name_exp"] == "sarcos":
        if params["model_type"] == 'mlp':
            hnet = HyperMLPMlp(params["hidden_dim"])
        elif params["model_type"] == 'trans_relu':
            hnet = HyperMLPTrans(params["hidden_dim"],act_type='relu')
        else:
            hnet = HyperMLPTrans(params["hidden_dim"],act_type='gelu')
        net = MLPTarget()
        return hnet, net
        