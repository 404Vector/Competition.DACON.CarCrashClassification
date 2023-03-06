import os
import random
import numpy as np
import torch

class SeedFixer:
    @staticmethod
    def seed_everything(seed:int = 21):
        # seed 고정
        print('Seed has been fixed :: {}'.format(seed))
        random_seed = seed
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True # type: ignore    
        torch.backends.cudnn.benchmark = False # type: ignore    
        np.random.seed(random_seed)
        random.seed(random_seed)
        os.environ['PYTHONHASHSEED'] = str(random_seed)