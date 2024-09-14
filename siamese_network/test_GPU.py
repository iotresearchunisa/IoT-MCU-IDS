"""
### PyTorch and CUDA Environment Checker

This script utilizes the PyTorch library to display information about the current PyTorch installation and the system's GPU capabilities.
Specifically, it:

1. Prints the installed PyTorch version.
2. Checks if CUDA (NVIDIA's parallel computing platform) is available.
3. Displays the number of GPUs detected by PyTorch.
4. If CUDA is available, it prints the name of the first GPU. Otherwise, it notifies that no GPU was detected.

"""

import torch

print("PyTorch version:", torch.__version__)

cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

gpu_count = torch.cuda.device_count()
print("Number of GPUs:", gpu_count)

if cuda_available:
    gpu_name = torch.cuda.get_device_name(0)
    print("GPU Name:", gpu_name)
else:
    print("No GPU detected.")
