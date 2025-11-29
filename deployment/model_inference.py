import torch
import torch.nn as nn
import numpy as np
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_HOME'] = ''
os.environ['CUDA_LIB'] = ''

torch.set_default_device('cpu')

_original_cuda_is_available = torch.cuda.is_available
torch.cuda.is_available = lambda: False

try:
    from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, sLSTMBlockConfig, mLSTMLayerConfig, sLSTMLayerConfig, FeedForwardConfig
    
    try:
        from xlstm.blocks.slstm import cell as slstm_cell_module
        
        _original_slstm_new = slstm_cell_module.sLSTMCell.__new__
        def patched_slstm_new(cls, config, skip_backend_init=False):
            return slstm_cell_module.sLSTMCell_vanilla(config, skip_backend_init=skip_backend_init)
        slstm_cell_module.sLSTMCell.__new__ = staticmethod(patched_slstm_new)
        
        print("✓ xlstm patched to use CPU-only (vanilla) backend for sLSTM")
    except Exception as e:
        print(f"Warning: Could not patch xlstm backend: {e}")
        
except ImportError:
    print("Warning: xlstm not found. Please install it with: pip install xlstm")
    raise

class ParallelxLSTMClassifierInference(nn.Module):
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

def load_model_from_checkpoint(checkpoint_path, config):
    torch.set_default_device('cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    with torch.device('cpu'):
        model = ParallelxLSTMClassifierInference(config)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    try:
        model_dict = model.state_dict()
        
        adjusted_state_dict = {}
        for k, v in state_dict.items():
            if k in model_dict:
                if '_recurrent_kernel_' in k and v.shape != model_dict[k].shape:
                    if len(v.shape) == 3 and len(model_dict[k].shape) == 3:
                        adjusted_state_dict[k] = v.transpose(1, 2).contiguous()
                        print(f"  Adjusted shape for {k}: {v.shape} -> {adjusted_state_dict[k].shape}")
                    else:
                        adjusted_state_dict[k] = v
                elif v.shape == model_dict[k].shape:
                    adjusted_state_dict[k] = v
                else:
                    print(f"  Skipping {k}: shape mismatch {v.shape} vs {model_dict[k].shape}")
            else:
                print(f"  Skipping {k}: not in model")
        
        model.load_state_dict(adjusted_state_dict, strict=False)
        print(f"✓ Loaded {len(adjusted_state_dict)}/{len(state_dict)} weights")
    except Exception as e:
        print(f"Warning: Error loading weights: {e}")
    
    model.eval()
    model.to('cpu')
    
    return model
