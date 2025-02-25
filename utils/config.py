import numpy as np
import torch
import random
import os

BASE_PATH  = '/kaggle/input/uw-madison-gi-tract-image-segmentation'

class CFG:

    def __init__(self,args) -> None:

        self.seed          = args.seed
        self.debug         = False if args.debug == "False" else True # set debug=False for Full Training
        self.two_half_D    = False if args.two_half_D == "False" else True
        self.exp_name      = args.exp_name
        self.comment       = args.comment
        self.model_name    = args.model_name
        self.backbone      = args.backbone if not self.two_half_D else "timm-efficientnet-b2"
        self.train_bs      = args.train_bs if not self.two_half_D else 64
        self.valid_bs      = self.train_bs*2
        self.img_size      = [224, 224] if not self.two_half_D else [160, 192]
        self.epochs        = args.epochs if not self.two_half_D else 5
        self.lr            = args.lr
        self.scheduler     = args.scheduler
        self.min_lr        = 1e-6
        self.T_max         = int(30000/self.train_bs*self.epochs)+50
        self.T_0           = 25
        self.warmup_epochs = 0
        self.wd            = 1e-6
        self.n_accumulate  = max(1, 32//self.train_bs)
        self.n_fold        = args.n_fold
        self.num_classes   = 3
        self.device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fold_no       = args.fold_no
    

    def display(self):
        print(f"{self.exp_name}")
        print(f"debug is {self.debug}")
        print(f"seed is {self.seed}")
        print(f"two_half_D is {self.two_half_D}")
        print(f"train_bs is {self.train_bs}")
        print(f"img_size is {self.img_size}")
        print(f"fold_no is {self.fold_no}")
        print(f"backbone is {self.backbone}")
        print(f"epochs is {self.epochs}")



def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    print(f"Setting seed as {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')

#this line will go in main or doesnt have to if it prints seeding done.

def initialise_config(args):
    cfg = CFG(args)
    set_seed(cfg.seed)
    return cfg

