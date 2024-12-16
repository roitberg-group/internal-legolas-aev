from importlib.metadata import version, PackageNotFoundError

import torch

from torchani import aev
from torchani import utils
from torchani.aev import AEVComputer

try:
    __version__ = version("torchani")
except PackageNotFoundError:
    pass  # package is not installed

__all__ = [
    "utils",
    "aev",
    "AEVComputer",
]

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    max_sm_major = max(
        [torch.cuda.get_device_capability(i)[0] for i in range(num_devices)]
    )
