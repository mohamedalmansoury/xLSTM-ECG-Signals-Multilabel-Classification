# Model Architecture Documentation

## Overview

This document provides detailed information about the Parallel xLSTM architecture used for ECG classification.

## Architecture Components

### 1. Input Processing

#### Signal Input
- **Shape**: `(batch_size, 1000, 12)`
- **1000 timesteps**: 10 seconds @ 100 Hz sampling rate
- **12 channels**: Standard 12-lead ECG (I, II, III, aVR, aVL, aVF, V1-V6)

#### Metadata Input
- **Shape**: `(batch_size, 2)`
- **Features**: [age, sex]
  - Age: Integer (0-120)
  - Sex: Binary (0=Female, 1=Male)

### 2. Embedding Layer

```python
embedding = Linear(12, 128)
embedding_norm = LayerNorm(128)
embedding_act = ReLU()
```

- Projects 12-lead signals to 128-dimensional space
- Layer normalization for training stability
- ReLU activation for non-linearity

### 3. Parallel xLSTM Branches

#### sLSTM Branch (Spatial)

**Purpose**: Capture spatial relationships between ECG leads

**Configuration**:
```python
sLSTMBlockConfig(
    slstm=sLSTMLayerConfig(
        num_heads=4,
        conv1d_kernel_size=4,
        bias_init="powerlaw_blockdependent"
    ),
    feedforward=FeedForwardConfig(
        proj_factor=1.3,
        act_fn="gelu"
    )
)
```

**Architecture**:
- 2 stacked sLSTM blocks
- 4 attention heads per block
- Conv1D with kernel size 4 for local context
- GELU activation in feedforward layers

**Key Features**:
- Multi-head attention for different spatial patterns
- Convolutional component for local lead interactions
- Power-law bias initialization for better convergence

#### mLSTM Branch (Memory)

**Purpose**: Capture temporal dependencies and long-range patterns

**Configuration**:
```python
mLSTMBlockConfig(
    mlstm=mLSTMLayerConfig(
        num_heads=4,
        conv1d_kernel_size=4
    )
)
```

**Architecture**:
- 2 stacked mLSTM blocks
- 4 attention heads per block
- Memory-augmented cells for long-term dependencies

**Key Features**:
- Enhanced memory capacity for temporal patterns
- Multi-head structure for diverse temporal features
- Efficient gradient flow for long sequences

### 4. Feature Fusion

```python
x_fused = (x_slstm + x_mlstm) / 2.0
x_avg = Mean(x_fused, dim=1)
x_max = Max(x_fused, dim=1)
x_pool = Concat([x_avg, x_max])
```

**Process**:
1. Average the sLSTM and mLSTM outputs
2. Apply dropout for regularization
3. Global average pooling across time
4. Global max pooling across time
5. Concatenate pooled features

**Output Shape**: `(batch_size, 256)`

### 5. Metadata Processing

```python
metadata_net = Sequential(
    Linear(2, 32),
    ReLU(),
    Dropout(0.3)
)
```

**Purpose**: Extract relevant features from patient demographics
**Output Shape**: `(batch_size, 32)`

### 6. Classification Head

```python
classifier = Sequential(
    Linear(288, 256),  # 256 from ECG + 32 from metadata
    ReLU(),
    Dropout(0.3),
    Linear(256, 128),
    ReLU(),
    Dropout(0.3),
    Linear(128, 5),
    Sigmoid()
)
```

**Architecture**:
- Three-layer MLP with decreasing dimensions
- ReLU activations for non-linearity
- Dropout (0.3) after each layer for regularization
- Sigmoid output for multilabel classification

**Output Shape**: `(batch_size, 5)` - Probabilities for each class

## Model Parameters

### Total Parameters
Approximately **15-20 million** trainable parameters

### Parameter Distribution
- Embedding layers: ~1.5K
- sLSTM branch: ~7-8M
- mLSTM branch: ~7-8M
- Metadata network: ~1K
- Classifier: ~75K

## Training Configuration

### Hyperparameters

```python
CONFIG = {
    'embedding_dim': 128,
    'slstm_hidden_size': 256,
    'mlstm_hidden_size': 256,
    'num_heads': 4,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 14,
    'patience': 3  # Early stopping
}
```

### Optimizer
- **Adam** optimizer
- Learning rate: 0.001
- Default beta parameters: (0.9, 0.999)

### Loss Function
- **Binary Cross-Entropy** for multilabel classification
- Applied per class independently

### Training Strategy
- Stratified train/validation/test split
- Early stopping with patience=3
- Model checkpoint saving (best validation F1)
- Learning rate scheduling (optional)

## Design Decisions

### Why Parallel Architecture?

1. **Complementary Features**:
   - sLSTM: Spatial inter-lead relationships
   - mLSTM: Temporal patterns within each lead

2. **Robustness**:
   - Redundancy improves generalization
   - Reduces overfitting to specific patterns

3. **Performance**:
   - Empirically better than single-branch models
   - Captures both spatial and temporal aspects

### Why xLSTM over Standard LSTM?

1. **Improved Memory**: Better long-term dependency modeling
2. **Attention Mechanism**: More flexible than fixed gates
3. **Scalability**: Better performance on longer sequences
4. **Training Stability**: Improved gradient flow

### Why Multilabel?

ECG abnormalities often co-occur:
- Patient can have multiple conditions simultaneously
- More clinically realistic than single-label
- Better utilization of dataset labels

## Inference Pipeline

```
Input Signal (1000, 12) + Metadata (2)
            ↓
    Embedding (1000, 128)
            ↓
    ┌───────┴───────┐
    ↓               ↓
sLSTM (1000,128)  mLSTM (1000,128)
    ↓               ↓
    └───────┬───────┘
            ↓
    Fusion (1000, 128)
            ↓
    Pooling (256)
            ↓        ↓
    ECG Features  Metadata Features
         (256)         (32)
            └─────┬─────┘
                  ↓
          Classification (5)
                  ↓
            Probabilities
        [NORM, MI, STTC, CD, HYP]
```

## Model Variants

### Considered Alternatives

1. **CNN-based**: Good spatial features, limited temporal modeling
2. **Transformer**: Excellent attention, high computational cost
3. **Standard LSTM**: Proven for sequences, limited memory capacity
4. **Hybrid CNN-LSTM**: Good balance, less sophisticated than xLSTM

### Future Improvements

1. **Attention Visualization**: Interpretability of important leads/segments
2. **Multi-scale Processing**: Different temporal resolutions
3. **Uncertainty Quantification**: Bayesian or ensemble approaches
4. **Transfer Learning**: Pre-training on larger ECG datasets
5. **Lightweight Variants**: Model compression for edge deployment

## References

1. Beck, M., et al. (2024). "xLSTM: Extended Long Short-Term Memory"
2. Vaswani, A., et al. (2017). "Attention Is All You Need"
3. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory"
4. Ribeiro, A. H., et al. (2020). "Automatic diagnosis of the 12-lead ECG using a deep neural network"

## License

MIT License - See [LICENSE](../LICENSE) for details.
