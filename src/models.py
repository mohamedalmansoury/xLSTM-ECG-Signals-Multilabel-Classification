import torch
import torch.nn as nn
from xlstm import (
    xLSTMBlockStack, 
    xLSTMBlockStackConfig, 
    mLSTMBlockConfig, 
    sLSTMBlockConfig, 
    mLSTMLayerConfig, 
    sLSTMLayerConfig, 
    FeedForwardConfig
)


class ParallelxLSTMClassifier(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Linear(config['input_shape'][1], config['embedding_dim'])
        self.embedding_norm = nn.LayerNorm(config['embedding_dim'])
        self.embedding_act = nn.ReLU()
        
        slstm_config = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                num_heads=config['num_heads'],
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(
                proj_factor=1.3,
                act_fn="gelu",
            ),
        )
        
        slstm_stack_config = xLSTMBlockStackConfig(
            slstm_block=slstm_config,
            context_length=config['input_shape'][0],
            num_blocks=2,
            embedding_dim=config['embedding_dim'],
            slstm_at=[0, 1],
        )
        self.slstm_branch = xLSTMBlockStack(slstm_stack_config)
        
        mlstm_config = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                num_heads=config['num_heads'],
                conv1d_kernel_size=4,
            ),
        )
        
        mlstm_stack_config = xLSTMBlockStackConfig(
            mlstm_block=mlstm_config,
            context_length=config['input_shape'][0],
            num_blocks=2,
            embedding_dim=config['embedding_dim'],
            slstm_at=[],
        )
        self.mlstm_branch = xLSTMBlockStack(mlstm_stack_config)
        
        self.dropout = nn.Dropout(config['dropout'])
        
        self.metadata_net = nn.Sequential(
            nn.Linear(config['metadata_dim'], 32),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        
        combined_dim = (config['embedding_dim'] * 2) + 32
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(128, config['num_classes']),
            nn.Sigmoid()
        )
        
    def forward(self, x, m):
        x = self.embedding(x)
        x = self.embedding_norm(x)
        x = self.embedding_act(x)
        
        x_s = self.slstm_branch(x)
        x_m = self.mlstm_branch(x)
        
        x_fused = (x_s + x_m) / 2.0
        x_fused = self.dropout(x_fused)
        
        x_avg = torch.mean(x_fused, dim=1)
        x_max, _ = torch.max(x_fused, dim=1)
        x_pool = torch.cat([x_avg, x_max], dim=1)
        
        m_out = self.metadata_net(m)
        
        combined = torch.cat([x_pool, m_out], dim=1)
        output = self.classifier(combined)
        
        return output
