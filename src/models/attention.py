"""
Dual-Channel Attention Mechanism for Information Fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class DualInfoTransformer(nn.Module):
    """
    Dual-channel attention module for fusing inbound and outbound flow information
    Uses multi-headed attention to capture cross-dependencies
    """
    
    def __init__(self, h=4, d_nodes=288, d_channel=512, d_model=96):
        super(DualInfoTransformer, self).__init__()
        
        assert d_model % h == 0, "d_model must be divisible by number of heads"
        
        self.d_nodes = d_nodes
        self.d_model = d_model
        self.d_channel = d_channel
        self.d_k = d_channel // h
        self.h = h
        
        # Query and value transformations for OD flow
        self.od_linears = self._clone_modules(
            nn.Sequential(
                nn.Conv1d(d_model, d_channel, kernel_size=1),
                nn.PReLU(d_channel),
                nn.Conv1d(d_channel, d_channel, kernel_size=1),
                nn.PReLU(d_channel)
            ), 2
        )
        
        # Key transformation for outbound flow
        self.od_out_linears = nn.Sequential(
            nn.Conv1d(d_model, d_channel, kernel_size=1),
            nn.PReLU(d_channel),
            nn.Conv1d(d_channel, d_channel, kernel_size=1),
            nn.PReLU(d_channel)
        )
        
        # Output projection layers
        self.od_conv = nn.Sequential(
            nn.Conv1d(d_channel, d_channel, kernel_size=1),
            nn.PReLU(d_channel),
            nn.Conv1d(d_channel, d_model, kernel_size=1),
            nn.PReLU(d_model)
        )
    
    def _clone_modules(self, module, N):
        """Create N identical copies of a module"""
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
    def _attention(self, query, key, value):
        """
        Compute scaled dot-product attention
        
        Args:
            query, key, value: [batch, heads, nodes, d_k]
        
        Returns:
            attended: [batch, heads, nodes, d_k]
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value)
    
    def _multi_head_attention(self, hid_od, hid_od_out):
        """
        Apply multi-headed attention between OD and outbound flows
        
        Args:
            hid_od: Hidden states for OD flow [batch, nodes, d_model]
            hid_od_out: Hidden states for outbound flow [batch, nodes, d_model]
        
        Returns:
            attn_od: Attention-weighted OD features [batch, nodes, d_model]
        """
        # Permute for convolution
        hid_od = hid_od.permute(0, 2, 1)
        hid_od_out = hid_od_out.permute(0, 2, 1)
        
        nbatches = hid_od.size(0)
        
        # Linear projections in batch
        odquery, odvalue = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.od_linears, (hid_od, hid_od))
        ]
        
        # Key from outbound flow
        od_out_key = self.od_out_linears(hid_od_out).view(
            nbatches, -1, self.h, self.d_k
        ).transpose(1, 2)
        
        # Apply attention
        attn_od = self.od_conv(
            self._attention(
                query=odquery,
                key=od_out_key,
                value=odvalue
            ).transpose(-2, -1).contiguous().view(-1, self.d_channel, self.d_nodes)
        )
        
        attn_od = attn_od.permute(0, 2, 1)
        return attn_od
    
    def forward(self, hidden_states_od, hidden_states_od_out):
        """
        Forward pass through dual-channel attention
        
        Args:
            hidden_states_od: OD flow hidden states
            hidden_states_od_out: Outbound flow hidden states
        
        Returns:
            Fused features through attention mechanism
        """
        return self._multi_head_attention(hidden_states_od, hidden_states_od_out)
