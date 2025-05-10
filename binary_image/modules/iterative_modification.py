
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from binary_image.image import BinaryImage

class InterativeModifier(nn.Module):
    def __init__(self, neighbor_type=1):
        super(InterativeModifier, self).__init__()
        self.neighbor_type = neighbor_type
        
    def euler_diff(self, x):
        euler_before = BinaryImage(x).euler_number
        x_after = x.clone()
        x_after[:, 1, 1] = 1 - x_after[:, 1, 1]
        euler_after = BinaryImage(x_after).euler_number
        return euler_after - euler_before
    
    def modify(self, x, operation):
        diff = self.euler_diff(x)
        aij = diff == self.neighbor_type
        bij = x[:, 1, 1]
        
        if not aij and not bij:
            target = operation[0]
        elif not aij and bij:
            target = operation[1]
        elif aij and not bij:
            target = operation[2]  
        else:
            target = operation[3]
        
        if x[:, 1, 1] != target:
            print(f"Operation: {operation}")
            print(f"Euler diff: {diff}")
            print(f"Current pixel value: {x[:, 1, 1]}")
            print(f"Target pixel value: {target}")
            print(f"Condition aij: {aij}, bij: {bij}")            

        x[:, 1, 1] = target
        
        return x
        
    def forward(self, x, operation) -> BinaryImage:
        H, W = x.shape[1], x.shape[2]
        x = F.pad(x, (1, 1, 1, 1), mode='constant', value=0)
        for i in range(H):
            for j in range(W):
                sc, sr = (i, j)
                ec, er = (i + 3, j + 3)
                x_patch = x[:, sc:ec, sr:er]
                x[:, sc:ec, sr:er] = self.modify(x_patch, operation=operation)
        x = x[:, 1:-1, 1:-1]
        return x