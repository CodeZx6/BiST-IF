"""
Evaluation Metrics for OD Flow Prediction
"""
import numpy as np


def masked_mape(y_pred, y_true, null_val=0.0):
    """
    Masked Mean Absolute Percentage Error
    Ignores locations where true value equals null_val
    """
    idx = np.nonzero(y_true != null_val)
    return np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx]))


def get_mae(pred, real):
    """Mean Absolute Error"""
    return np.mean(np.abs(real - pred))


def get_mse(pred, real):
    """Mean Squared Error"""
    return np.mean(np.power(real - pred, 2))


def get_rmse(pred, real):
    """Root Mean Squared Error"""
    return np.sqrt(get_mse(pred=pred, real=real))


def compute_metrics(predictions, targets):
    """
    Compute all evaluation metrics
    
    Args:
        predictions: Model predictions array
        targets: Ground truth array
    
    Returns:
        Dictionary containing MAE, RMSE, MAPE
    """
    mae = get_mae(predictions, targets)
    rmse = get_rmse(predictions, targets)
    mape = masked_mape(predictions, targets)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def compute_metrics_by_horizon(predictions, targets, horizons):
    """
    Compute metrics for each prediction horizon
    
    Args:
        predictions: [samples, horizons, nodes, nodes]
        targets: [samples, horizons, nodes, nodes]
        horizons: Number of horizons to evaluate
    
    Returns:
        Dictionary with metrics for each horizon
    """
    results = {}
    
    for h in range(horizons):
        y_pred = predictions[:, h, :, :]
        y_true = targets[:, h, :, :]
        
        metrics = compute_metrics(y_pred, y_true)
        results[f'horizon_{h+1}'] = metrics
    
    return results
