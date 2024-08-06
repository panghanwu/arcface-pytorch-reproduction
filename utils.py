import random
import numpy as np
import torch
from pathlib import Path

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_log_dir(name: str, parent: str = '') -> Path:
        parent = Path(parent)
        log_dir = parent / name
        index = 0
        while log_dir.exists():
            index += 1
            log_dir = parent / f'{name}-{index}'
        log_dir.mkdir(parents=True)
        return log_dir