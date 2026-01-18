"""
Complete BiST-IF Model Architecture
Spatio-Temporal Origin-Destination Prediction Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .od_network import ODNet
from .attention import DualInfoTransformer


def spatial_aggregation_h(F_tensor):
    """Spatial aggregation along horizontal dimension"""
    assert F_tensor.dim() == 4
    spatial_sum = F_tensor.sum(3, keepdim=True)
    return spatial_sum / F_tensor.size(3)


class ConvTemporalBlock(nn.Module):
    """
    Convolutional block for temporal feature extraction
    """
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvTemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=(1, kernel_size), 
                               stride=(1, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=(1, 1), 
                               stride=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        return out


class BiSTIF(nn.Module):
    """
    Main BiST-IF Architecture
    Multi-temporal spatio-temporal network for OD flow prediction
    """
    
    def __init__(self, device, num_nodes=80, hidden_dim=96, output_dim=80,
                 num_layers=2, seq_len=6, horizon=6, num_relations=1, K=2):
        super(BiSTIF, self).__init__()
        
        self.device = device
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_relations = num_relations
        self.K = K
        self.global_step = 0
        
        # OD Network component
        self.OD = ODNet(
            num_nodes=num_nodes,
            num_output_dim=output_dim,
            num_units=hidden_dim,
            num_finished_input_dim=num_nodes,
            num_rnn_layers=num_layers,
            seq_len=seq_len,
            horizon=horizon,
            num_relations=num_relations,
            K=K
        )
        
        # Temporal convolutional modules
        self.week_conv = ConvTemporalBlock(1, 256, kernel_size=12)
        self.day_conv = ConvTemporalBlock(1, 256, kernel_size=6)
        self.hour_conv = ConvTemporalBlock(1, 256, kernel_size=12)
        
        # Attention fusion module
        self.attention_fusion = DualInfoTransformer(
            h=4,
            d_nodes=num_nodes,
            d_channel=512,
            d_model=hidden_dim
        )
        
        # Activation function
        self.activation = nn.PReLU()
    
    def _encoder_od_od_out(self, x_od, x_od_out):
        """
        Encode OD and outbound flow sequences
        
        Args:
            x_od: OD flow sequence [batch, time, nodes, nodes]
            x_od_out: Outbound flow sequence [batch, time, nodes, nodes]
        
        Returns:
            enc_hiddens_od: List of hidden states from all layers
        """
        batch_size = x_od.size(0)
        seq_len = x_od.size(1)
        
        # Initialize hidden states
        enc_hiddens_od = [
            torch.zeros(batch_size, self.num_nodes, self.hidden_dim,
                       dtype=x_od.dtype, device=x_od.device)
            for _ in range(self.num_layers)
        ]
        
        # Encode sequence
        for t in range(seq_len):
            x_od_t = x_od[:, t, :, :]
            
            # First layer
            enc_first_out_od = self.OD.encoder_first_layer(
                x_od_t, 
                enc_hiddens_od[0]
            )
            enc_hiddens_od[0] = enc_first_out_od
            enc_mid_out_od = enc_first_out_od
            
            # Additional layers
            for idx in range(self.num_layers - 1):
                enc_mid_out_od = self.activation(enc_mid_out_od)
                enc_mid_out_od = self.OD.encoder_second_layer(
                    idx,
                    enc_mid_out_od,
                    enc_hiddens_od[idx + 1]
                )
                enc_hiddens_od[idx + 1] = enc_mid_out_od
        
        return enc_hiddens_od
    
    def _decoder_od_od_out(self, sequences_y, enc_hiddens_od):
        """
        Decode to predict future OD flows
        
        Args:
            sequences_y: Target sequences (Y_od, Y_od_out)
            enc_hiddens_od: Encoded hidden states
        
        Returns:
            predictions_od: Predicted OD flows [batch, horizon, nodes, nodes]
        """
        predictions_od = []
        
        Y_od, Y_od_out = sequences_y[0], sequences_y[1]
        
        # Initialize decoder input
        GO_od = torch.zeros(enc_hiddens_od[0].size()[0],
                           enc_hiddens_od[0].size()[1],
                           self.output_dim,
                           dtype=enc_hiddens_od[0].dtype,
                           device=enc_hiddens_od[0].device)
        
        dec_input_od = GO_od
        dec_hiddens_od = enc_hiddens_od
        
        # Decode sequence
        for t in range(self.horizon):
            y_od, y_od_out = Y_od[:, t, :, :], Y_od_out[:, t, :, :]
            
            # First decoder layer
            dec_first_out_od = self.OD.decoder_first_layer(
                dec_input_od,
                dec_hiddens_od[0]
            )
            dec_hiddens_od[0] = dec_first_out_od
            dec_mid_out_od = dec_first_out_od
            
            # Additional decoder layers
            for idx in range(self.num_layers - 1):
                dec_mid_out_od = self.activation(dec_mid_out_od)
                dec_mid_out_od = self.OD.decoder_second_layer(
                    idx,
                    dec_mid_out_od,
                    dec_hiddens_od[idx + 1]
                )
                dec_hiddens_od[idx + 1] = dec_mid_out_od
            
            # Output projection
            dec_mid_out_od = self.OD.output_layer(dec_mid_out_od)
            predictions_od.append(dec_mid_out_od)
            
            # Scheduled sampling
            dec_input_od = self._scheduled_sampling(
                dec_mid_out_od,
                y_od,
                GO_od
            )
        
        if self.training:
            self.global_step += 1
        
        return torch.stack(predictions_od).transpose(0, 1)
    
    def _scheduled_sampling(self, out, label, GO):
        """
        Scheduled sampling strategy
        During training, mix predicted and ground truth values
        """
        use_truth_sequence = False
        
        if use_truth_sequence:
            decoder_input = label
        else:
            decoder_input = out.detach().view(-1, self.num_nodes, self.output_dim)
        
        return decoder_input
    
    def _temporal_module(self, x_week, x_day, x_hour):
        """
        Process multi-temporal patterns through CNNs
        
        Args:
            x_week: Weekly patterns [batch, nodes, nodes, time]
            x_day: Daily patterns [batch, nodes, nodes, time]
            x_hour: Hourly patterns [batch, nodes, nodes, time]
        
        Returns:
            Temporal features for each scale
        """
        # Add channel dimension
        x_week = x_week.unsqueeze(1)
        x_day = x_day.unsqueeze(1)
        x_hour = x_hour.unsqueeze(1)
        
        # Apply temporal convolutions
        week_feat = self.week_conv(x_week)
        day_feat = self.day_conv(x_day)
        hour_feat = self.hour_conv(x_hour)
        
        # Remove extra dimensions
        week_feat = week_feat.squeeze(-1)
        day_feat = day_feat.squeeze(-1)
        hour_feat = hour_feat.squeeze(-1)
        
        return week_feat, day_feat, hour_feat
    
    def forward(self, sequences, sequences_y):
        """
        Forward pass through STODP-Net
        
        Args:
            sequences: Input sequences (x_week_od, x_day_od, x_hour_od, x_week_od_out)
            sequences_y: Target sequences (y_od, y_od_out)
        
        Returns:
            predictions_od: Predicted OD flows
        """
        x_week_od = sequences[0]
        x_day_od = sequences[1]
        x_hour_od = sequences[2]
        x_week_od_out = sequences[3]
        
        # RNN-based prediction
        predictions_od_rnn = self._rnn_prediction(
            x_week_od[:, -6:, :, :],
            x_week_od_out[:, -12:, :, :],
            sequences_y
        )
        
        # Temporal CNN features
        x_week_od = x_week_od.permute(0, 2, 3, 1)
        x_day_od = x_day_od.permute(0, 2, 3, 1)
        x_hour_od = x_hour_od.permute(0, 2, 3, 1)
        
        week_feat, day_feat, hour_feat = self._temporal_module(
            x_week_od, x_day_od, x_hour_od
        )
        
        # Permute back
        week_feat = week_feat.permute(0, 3, 1, 2)
        day_feat = day_feat.permute(0, 3, 1, 2)
        hour_feat = hour_feat.permute(0, 3, 1, 2)
        
        # Combine predictions
        predictions_od = predictions_od_rnn + week_feat + day_feat + hour_feat
        
        return predictions_od
    
    def _rnn_prediction(self, x_week_od, x_week_od_out, sequences_y):
        """RNN-based prediction component"""
        enc_hiddens_od = self._encoder_od_od_out(x_week_od, x_week_od_out)
        predictions_od = self._decoder_od_od_out(sequences_y, enc_hiddens_od)
        return predictions_od
