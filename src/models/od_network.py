"""
Encoder-Decoder Network for OD Flow Modeling
"""
import torch
import torch.nn as nn
from .gru_cell import GGRUCell


class ODNet(nn.Module):
    """
    Origin-Destination Network with Encoder-Decoder architecture
    Processes OD matrices through stacked GRU cells
    """
    
    def __init__(self, num_nodes=80, num_output_dim=80, num_units=96,
                 num_finished_input_dim=80, num_rnn_layers=2, 
                 seq_len=6, horizon=6, num_relations=1, K=2, 
                 num_bases=1, dropout_type=None, dropout_prob=0.0,
                 global_fusion=False):
        super(ODNet, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_output_dim = num_output_dim
        self.num_units = num_units
        self.num_finished_input_dim = num_finished_input_dim
        self.num_rnn_layers = num_rnn_layers
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_relations = num_relations
        self.K = K
        self.num_bases = num_bases
        self.dropout_type = dropout_type
        self.dropout_prob = dropout_prob
        self.global_fusion = global_fusion
        
        # First encoder layer
        self.encoder_first_finished_cells = GGRUCell(
            self.num_finished_input_dim,
            self.num_units,
            self.dropout_type,
            self.dropout_prob,
            self.num_relations,
            num_bases=self.num_bases,
            K=self.K,
            num_nodes=self.num_nodes,
            global_fusion=self.global_fusion
        )
        
        # Additional encoder layers
        self.encoder_second_cells = nn.ModuleList([
            GGRUCell(
                self.num_units,
                self.num_units,
                self.dropout_type,
                self.dropout_prob,
                self.num_relations,
                num_bases=self.num_bases,
                K=self.K,
                num_nodes=self.num_nodes,
                global_fusion=self.global_fusion
            ) for _ in range(self.num_rnn_layers - 1)
        ])
        
        # First decoder layer
        self.decoder_first_cells = GGRUCell(
            self.num_finished_input_dim,
            self.num_units,
            self.dropout_type,
            self.dropout_prob,
            self.num_relations,
            num_bases=self.num_bases,
            K=self.K,
            num_nodes=self.num_nodes,
            global_fusion=self.global_fusion
        )
        
        # Additional decoder layers
        self.decoder_second_cells = nn.ModuleList([
            GGRUCell(
                self.num_units,
                self.num_units,
                self.dropout_type,
                self.dropout_prob,
                self.num_relations,
                self.K,
                num_nodes=self.num_nodes,
                global_fusion=self.global_fusion
            ) for _ in range(self.num_rnn_layers - 1)
        ])
        
        # Output projection layer
        self.output_layer = nn.Linear(self.num_units, self.num_output_dim)
    
    def encoder_first_layer(self, x_od, finished_hidden):
        """First encoder layer processing"""
        finished_out = self.encoder_first_finished_cells(
            inputs=x_od, 
            hidden=finished_hidden
        )
        return finished_out
    
    def encoder_second_layer(self, index, first_out, enc_second_hidden):
        """Subsequent encoder layer processing"""
        enc_second_out = self.encoder_second_cells[index](
            inputs=first_out,
            hidden=enc_second_hidden
        )
        return enc_second_out
    
    def decoder_first_layer(self, decoder_input, dec_first_hidden):
        """First decoder layer processing"""
        dec_first_out = self.decoder_first_cells(
            inputs=decoder_input,
            hidden=dec_first_hidden
        )
        return dec_first_out
    
    def decoder_second_layer(self, index, decoder_first_out, dec_second_hidden):
        """Subsequent decoder layer processing"""
        dec_second_out = self.decoder_second_cells[index](
            inputs=decoder_first_out,
            hidden=dec_second_hidden
        )
        return dec_second_out
