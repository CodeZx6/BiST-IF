"""
Data Loading and Preprocessing Module
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class DataLoader:
    """
    Custom data loader for batching temporal sequences
    """
    
    def __init__(self, x_week, x_day, x_hour, x_out_week, 
                 y_od, y_od_out, batch_size, shuffle=False):
        self.x_week = x_week
        self.x_day = x_day
        self.x_hour = x_hour
        self.x_out_week = x_out_week
        self.y_od = y_od
        self.y_od_out = y_od_out
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(x_week)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self.indices = np.arange(self.num_samples)
        
    def shuffle_data(self):
        """Shuffle data indices"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def get_iterator(self):
        """
        Create iterator for batching
        
        Yields:
            Batched data tuples
        """
        if self.shuffle:
            self.shuffle_data()
        
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, self.num_samples)
            batch_indices = self.indices[start_idx:end_idx]
            
            yield (
                self.x_week[batch_indices],
                self.x_day[batch_indices],
                self.x_hour[batch_indices],
                self.x_out_week[batch_indices],
                self.y_od[batch_indices],
                self.y_od_out[batch_indices]
            )


def get_sample_indices(metro_edge_matrix, metro_edge_matrix_out,
                       time_matrix, raw_od_out_distribution,
                       raw_edge_in_matrix1, raw_edge_in_matrix2,
                       raw_edge_in_matrix3, raw_edge_in_matrix4,
                       raw_edge_in_matrix5, raw_edge_in_matrix6,
                       raw_flow_in_day_delay_expand1,
                       raw_flow_in_day_delay_expand2,
                       raw_flow_in_day_delay_expand3,
                       raw_flow_in_day_delay_expand4,
                       raw_flow_in_day_delay_expand5,
                       raw_flow_in_day_delay_expand6,
                       raw_od_distribution1, raw_od_distribution2,
                       raw_od_distribution3, raw_od_distribution4,
                       raw_od_distribution5, raw_od_distribution6,
                       num_of_weeks, num_of_days, num_of_hours,
                       idx, num_for_predict, points_per_hour):
    """
    Extract a single training sample with multi-temporal windows
    
    Args:
        Various input data matrices and parameters
        idx: Current time index
        num_for_predict: Number of future steps to predict
        points_per_hour: Data points per hour
    
    Returns:
        Tuple of input and target sequences, or None if insufficient data
    """
    week_sample_end = idx + 1
    week_sample_start = week_sample_end - num_of_weeks * 7 * 24 * points_per_hour
    
    day_sample_end = idx + 1
    day_sample_start = day_sample_end - num_of_days * 24 * points_per_hour
    
    hour_sample_end = idx + 1
    hour_sample_start = hour_sample_end - num_of_hours * points_per_hour
    
    predict_start = idx + 1
    predict_end = predict_start + num_for_predict
    
    # Check boundary conditions
    if week_sample_start < 0:
        return None
    if predict_end > metro_edge_matrix.shape[0]:
        return None
    
    # Extract sequences
    x_od_week = metro_edge_matrix[week_sample_start:week_sample_end]
    x_od_out_week = metro_edge_matrix_out[week_sample_start:week_sample_end]
    x_od_day = metro_edge_matrix[day_sample_start:day_sample_end]
    x_od_hour = metro_edge_matrix[hour_sample_start:hour_sample_end]
    
    # Future targets
    y_od = metro_edge_matrix[predict_start:predict_end]
    y_od_out = metro_edge_matrix_out[predict_start:predict_end]
    
    return x_od_week, x_od_out_week, x_od_day, x_od_hour, None, y_od, y_od_out
