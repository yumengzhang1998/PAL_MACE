import openmm
from openmmtorch import TorchForce
import torch

print("OpenMM:", openmm.__version__)
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("TorchForce OK:", TorchForce)
