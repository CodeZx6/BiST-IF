"""
Data Normalization and Scaling Utilities
"""
import numpy as np
import torch


class StandardScaler:
    """
    Standard scaler for normalizing data
    Computes mean and std for normalization
    """
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def transform(self, data):
        """Normalize data using mean and std"""
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        """Denormalize data back to original scale"""
        return (data * self.std) + self.mean


class StandardScalerTorch:
    """
    PyTorch version of standard scaler
    Operates on GPU tensors efficiently
    """
    
    def __init__(self, mean, std, device):
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(std, dtype=torch.float32, device=device)
        self.device = device
    
    def transform(self, data):
        """Normalize PyTorch tensor"""
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        """Denormalize PyTorch tensor"""
        return (data * self.std) + self.mean


def compute_scaler_params(data, axis):
    """
    Compute mean and std for normalization
    
    Args:
        data: Input data array
        axis: Axes along which to compute statistics
    
    Returns:
        mean, std: Statistics for normalization
    """
    mean = data.mean(axis=axis)
    std = data.std(axis=axis)
    
    # Avoid division by zero
    std = np.where(std == 0, 1.0, std)
    
    return mean, std
