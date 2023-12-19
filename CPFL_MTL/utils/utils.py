import json
import logging
import random
from pathlib import Path
import numpy as np
import torch
import os
from pymoo.factory import get_reference_directions
from pymoo.factory import get_performance_indicator
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def hypervolumn(A, ref):
    hv = get_performance_indicator("hv",ref_point=ref)
    return hv.do(A)

def get_test_rays(num_tasks, n_partitions=None, scaling=None, name_alg = "das-dennis"):
    """Create num_ray_eval rays for evaluation. Not pretty but does the trick"""
    if scaling != None:
        test_rays = get_reference_directions(name_alg, num_tasks, n_partitions=6, scaling=0.5).astype(
            np.float32
        )
    else:
        test_rays = get_reference_directions(name_alg, num_tasks,n_partitions=n_partitions).astype(
            np.float32
        )
    
    test_rays = test_rays[[(r > 0).all() for r in test_rays]][5:-5:2]
    test_rays = np.array(test_rays)
    print("Generated rays shape: ",test_rays.shape)
    return test_rays

def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


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

class ReDirectSTD(object):
    """
    overwrites the sys.stdout or sys.stderr
    Args:
      fpath: file cam_path
      console: one of ['stdout', 'stderr']
      immediately_visiable: False
    Usage example:
      ReDirectSTD('stdout.txt', 'stdout', False)
      ReDirectSTD('stderr.txt', 'stderr', False)
    """

    def __init__(self, fpath=None, console='stdout', immediately_visiable=False):
        import sys
        import os
        assert console in ['stdout', 'stderr']
        self.console = sys.stdout if console == "stdout" else sys.stderr
        self.file = fpath
        self.f = None
        self.immediately_visiable = immediately_visiable

        if fpath is not None:
            # Remove existing log file
            if os.path.exists(fpath):
                os.remove(fpath)
        if console == 'stdout':
            sys.stdout = self
        else:
            sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, **args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            if not os.path.exists(os.path.dirname(os.path.abspath(self.file))):
                os.mkdir(os.path.dirname(os.path.abspath(self.file)))

            if self.immediately_visiable:
                # open for writing, appending to the end of the file if it exists
                with open(self.file, 'a') as f:
                    f.write(msg)
            else:
                if self.f is None:
                    self.f = open(self.file, 'w')

                # print("self.f is not none")
                # first time self.f is None, second is not None
                self.f.write(msg)

    def flush(self):
        self.console.flush()
        if self.f is not None:
            self.f.flush()
            import os
            os.fsync(self.f.fileno())

    def close(self):
        self.console.close()
        if self.f is not None:
            self.f.close()



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


def save_args(folder, args, name="config.json", check_exists=False):
    set_logger()
    path = Path(folder)
    if check_exists:
        if path.exists():
            logging.warning(f"folder {folder} already exists! old files might be lost.")
    path.mkdir(parents=True, exist_ok=True)

    json.dump(vars(args), open(path / name, "w"))


def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = 1e-6 if min_angle is None else min_angle
    ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]
