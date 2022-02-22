from conf import *
from loader import *
from models import *
from trainer import *
from loss import *
from scheduler import *

import torch
import os
import numpy as np

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def main():
    set_seed(args.seed)


if __name__ == '__main__':
    main()