import os
import torch
import collections
import numpy as np
import json
import random

specifications_filename = "params.json"

def seed_everything(seed=42):
    # Python random
    random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # NumPy
    np.random.seed(seed)

    # Open3D
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)  # Set Open3D verbosity level
    # o3d.utility.set_random_seed(seed)

    # CuDNN (if available)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Other random number generators (adjust as needed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.default_rng(seed)

def load_experiment_specifications(experiment_directory):

    filename = os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"params.json"'.format(experiment_directory)
        )

    return json.load(open(filename))


def dict_to_device(ob, device):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_device(v, device) for k, v in ob.items()}
    else:
        return ob.to(device)


def to_numpy(x):
    return x.detach().cpu().numpy()


def to_torch(x, device='cpu'):
    if isinstance(x, list):
        return torch.Tensor(x).float().to(device)
    else:
        return torch.from_numpy(x).float().to(device)
