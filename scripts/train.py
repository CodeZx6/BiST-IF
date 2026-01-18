"""
Model Training Script
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import yaml
import argparse
from datetime import datetime

from src.models.BiSTIF import BiSTIF
from src.utils.metrics import compute_metrics, compute_metrics_by_horizon
from src.utils.scaler import StandardScaler, StandardScalerTorch
from src.data.dataloader import DataLoader


def seed_everything(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class StepLRWithMinLR(torch.optim.lr_scheduler.MultiStepLR):
    """Learning rate scheduler with minimum LR constraint"""
    
    def __init__(self, optimizer, milestones, gamma=0.1, 
                 last_epoch=-1, min_lr=2.0e-6):
        self.min_lr = min_lr
        super().__init__(optimizer, milestones, gamma, last_epoch)
    
    def get_lr(self):
        lr_candidate = super().get_lr()
        if isinstance(lr_candidate, list):
            return [max(self.min_lr, lr) for lr in lr_candidate]
        else:
            return max(self.min_lr, lr_candidate)


def evaluate_model(model, data_loader, scaler, device, criterion):
    """
    Evaluate model on validation/test set
    
    Args:
        model: Trained model
        data_loader: DataLoader instance
        scaler: StandardScalerTorch for denormalization
        device: Computation device
        criterion: Loss function
    
    Returns:
        avg_loss, mae, rmse: Evaluation metrics
    """
    model.eval()
    predictions = []
    targets = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_data in data_loader.get_iterator():
            x_week, x_day, x_hour, x_out_week, y_od, y_od_out = batch_data
            
            # Move to device
            x_week = torch.tensor(x_week, dtype=torch.float32, device=device)
            x_day = torch.tensor(x_day, dtype=torch.float32, device=device)
            x_hour = torch.tensor(x_hour, dtype=torch.float32, device=device)
            x_out_week = torch.tensor(x_out_week, dtype=torch.float32, device=device)
            y_od = torch.tensor(y_od, dtype=torch.float32, device=device)
            y_od_out = torch.tensor(y_od_out, dtype=torch.float32, device=device)
            
            # Forward pass
            y_pred = model(
                [x_week, x_day, x_hour, x_out_week],
                [y_od, y_od_out]
            )
            
            # Denormalize
            y_pred = scaler.inverse_transform(y_pred)
            y_od = scaler.inverse_transform(y_od)
            
            # Compute loss
            loss = criterion(y_pred, y_od)
            total_loss += loss.item()
            num_batches += 1
            
            # Store for metrics
            predictions.append(y_pred.cpu().numpy())
            targets.append(y_od.cpu().numpy())
    
    # Concatenate results
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets)
    avg_loss = total_loss / num_batches
    
    return avg_loss, metrics['MAE'], metrics['RMSE']


def train_epoch(model, data_loader, optimizer, scaler_torch, 
                criterion, device, max_grad_norm=5):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        data_loader: Training data loader
        optimizer: Optimizer
        scaler_torch: Scaler for denormalization
        criterion: Loss function
        device: Computation device
        max_grad_norm: Gradient clipping threshold
    
    Returns:
        avg_loss: Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    data_loader.shuffle_data()
    
    for batch_data in data_loader.get_iterator():
        x_week, x_day, x_hour, x_out_week, y_od, y_od_out = batch_data
        
        # Move to device
        x_week = torch.tensor(x_week, dtype=torch.float32, device=device)
        x_day = torch.tensor(x_day, dtype=torch.float32, device=device)
        x_hour = torch.tensor(x_hour, dtype=torch.float32, device=device)
        x_out_week = torch.tensor(x_out_week, dtype=torch.float32, device=device)
        y_od = torch.tensor(y_od, dtype=torch.float32, device=device)
        y_od_out = torch.tensor(y_od_out, dtype=torch.float32, device=device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(
            [x_week, x_day, x_hour, x_out_week],
            [y_od, y_od_out]
        )
        
        # Denormalize for loss computation
        y_pred = scaler_torch.inverse_transform(y_pred)
        y_od = scaler_torch.inverse_transform(y_od)
        
        # Compute loss
        loss = criterion(y_pred, y_od)
        
        # Backward pass
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def main(args):
    """Main training loop"""
    
    # Set random seed
    seed_everything(args.seed)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config['device'])
    
    # Create model
    model = BiSTIF(
        device=device,
        num_nodes=config['model']['num_nodes'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        num_layers=config['model']['num_layers'],
        seq_len=config['temporal']['seq_len'],
        horizon=config['temporal']['horizon']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        eps=1.0e-4
    )
    
    scheduler = StepLRWithMinLR(
        optimizer=optimizer,
        milestones=config['training']['scheduler']['milestones'],
        gamma=config['training']['scheduler']['gamma'],
        min_lr=config['training']['scheduler']['min_lr']
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    print("Starting training...")
    # Training loop would continue here...


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train STODP-Net')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    main(args)
