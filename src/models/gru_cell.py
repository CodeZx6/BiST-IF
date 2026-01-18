"""
Gated Recurrent Unit Cell for Spatio-Temporal Modeling
"""
import torch
import torch.nn as nn
from torch.nn import init, Parameter


class GGRUCell(nn.Module):
    """
    Graph-based Gated Recurrent Unit Cell
    Processes node-level sequential data with gating mechanisms
    """
    
    def __init__(self, in_channels, out_channels, dropout_type=None, 
                 dropout_prob=0.0, num_relations=3, num_bases=3, K=1, 
                 num_nodes=80, global_fusion=False):
        super(GGRUCell, self).__init__()
        
        self.num_chunks = 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_nodes = num_nodes
        self.global_fusion = global_fusion
        
        # Input transformation
        self.fc_gi = nn.Linear(in_channels, out_channels * self.num_chunks)
        # Hidden state transformation
        self.fc_gh = nn.Linear(out_channels, out_channels * self.num_chunks)
        
        # Gate biases
        self.bias_i = Parameter(torch.Tensor(self.out_channels))
        self.bias_r = Parameter(torch.Tensor(self.out_channels))
        self.bias_n = Parameter(torch.Tensor(self.out_channels))
        
        self.dropout_prob = dropout_prob
        self.dropout_type = dropout_type
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        init.ones_(self.bias_i)
        init.ones_(self.bias_r)
        init.ones_(self.bias_n)
    
    def forward(self, inputs, hidden=None):
        """
        Forward pass through GRU cell
        
        Args:
            inputs: Input tensor [batch, nodes, features]
            hidden: Previous hidden state [batch, nodes, features]
        
        Returns:
            output: Updated hidden state [batch, nodes, features]
        """
        if hidden is None:
            hidden = torch.zeros(inputs.size(0), inputs.size(1),
                                self.out_channels,
                                dtype=inputs.dtype,
                                device=inputs.device)
        
        # Transform inputs and hidden states
        gi = self.fc_gi(inputs)
        gh = self.fc_gh(hidden)
        
        # Split for gates
        i_i, i_n = gi.chunk(2, 2)
        h_i, h_n = gh.chunk(2, 2)
        
        # Input gate
        inputgate = torch.sigmoid(i_i + h_i + self.bias_i)
        # Candidate state
        newgate = torch.tanh(i_n + h_n + self.bias_n)
        # Update hidden state
        next_hidden = inputgate * newgate + (1 - newgate) * hidden
        
        return next_hidden
