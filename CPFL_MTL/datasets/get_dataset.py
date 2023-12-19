import os
print(os.getcwd())
from datasets.celebA import CELEBA
from datasets.mnist import MNIST
# from datasets.nlp import NLP
from datasets.nuyv2 import NYUv2
from datasets.sarcos import SARCOS

def get_dataset(params, configs):
    if configs["name_exp"] == "celebA":
        train_set = CELEBA(
            root=params["path"],
            is_transform=True,
            split="train",
            img_size=(params["img_rows"], params["img_cols"]),
            augmentations=None,
        )
        val_set = CELEBA(
            root=params["path"],
            is_transform=True,
            split="val",
            img_size=(params["img_rows"], params["img_cols"]),
            augmentations=None,
        )
        test_set = CELEBA(
            root=params["path"],
            is_transform=True,
            split="test",
            img_size=(params["img_rows"], params["img_cols"]),
            augmentations=None,
        )
        return train_set, val_set, test_set
    elif configs["name_exp"] == "mnist":
        train_set, val_set, test_set = MNIST(params["path"], params["val_size"])
        return train_set, val_set, test_set
    # elif params["dataset"] == "nlp":
    #     train_set, valid_set, test_set, embedding_matrix = NLP(params["path"])
    #     return train_set, valid_set, test_set, embedding_matrix
    elif configs["name_exp"] == "nuyv2":
        train_set = NYUv2(root=params['path'], train=True,mode = "train", augmentation=True)
        val_set = NYUv2(root=params['path'], mode = "val",train=False)
        test_set = NYUv2(root=params['path'], mode = "test",train=False)
        return train_set, valid_set, test_set
    elif configs["name_exp"] == "sarcos":
        train_set, valid_set, test_set = SARCOS(params['path'])
        return train_set, valid_set, test_set