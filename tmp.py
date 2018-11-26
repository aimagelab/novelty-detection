import torch
from glob import glob


for file in glob('checkpoints/cifar10/*.pkl'):
    checkpoint = torch.load(file)

    torch.save(checkpoint['weights'], file)