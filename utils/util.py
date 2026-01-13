import torch
import random
import numpy as np
import argparse
from PIL import ImageFilter, ImageOps

def write_log(log_file, str, mode='a'):
    with open(log_file, mode) as f:
        f.write(str)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform1, base_transform2=None):
        self.base_transform1 = base_transform1

        if base_transform2 == None:
            self.base_transform2 = base_transform1
        else :
            self.base_transform2 = base_transform2

    def __call__(self, x):
        q = self.base_transform1(x)
        k = self.base_transform2(x)
        return [q, k]
    
class NumSampleCropsTransform:
    """Take (2 x num_sample) random crops of one image as the query and key."""

    def __init__(self, base_transform1, base_transform2=None, num_sample=2):
        self.base_transform1 = base_transform1

        if base_transform2 == None:
            self.base_transform2 = base_transform1
        else :
            self.base_transform2 = base_transform2

        self.num_sample = num_sample

    def __call__(self, x):

        if isinstance(self.num_sample, int):
            if self.num_sample == 1:
                q = self.base_transform1(x)
                k = self.base_transform2(x)
            else:
                q, k = [], []
                for _ in range(self.num_sample):
                    q.append(self.base_transform1(x))
                    k.append(self.base_transform2(x))
                q, k = torch.stack(q), torch.stack(k)

        elif isinstance(self.num_sample, list):
            q, k = [], []
            for _ in range(self.num_sample[0]): q.append(self.base_transform1(x))
            for _ in range(self.num_sample[1]): k.append(self.base_transform2(x))
            q, k = torch.stack(q), torch.stack(k)

        return [q, k]

class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)