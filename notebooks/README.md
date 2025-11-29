# Notebooks

This directory contains Jupyter notebooks for training and experimentation.

## Main Notebook

### `ECG_Signals_Classification_xLSTM.ipynb`

Complete pipeline for training the xLSTM ECG classification model.

**Contents**:
1. **Data Loading**: PTB-XL dataset loading and exploration
2. **Preprocessing**: Signal filtering, normalization, and augmentation
3. **Model Definition**: Parallel xLSTM architecture
4. **Training**: PyTorch Lightning training loop
5. **Evaluation**: Metrics, confusion matrices, ROC curves
6. **Export**: Model checkpoint and normalization parameters

## Running the Notebook

### Local Jupyter

```bash
# Activate virtual environment
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Jupyter
pip install jupyter

# Launch Jupyter
jupyter notebook notebooks/ECG_Signals_Classification_xLSTM.ipynb
```

### Google Colab

1. Upload notebook to Google Drive
2. Open with Google Colab
3. Change runtime to GPU (Runtime → Change runtime type → GPU)
4. Upload or mount PTB-XL dataset
5. Run cells sequentially

### Kaggle Notebooks

1. Create new Kaggle notebook
2. Add PTB-XL dataset from Kaggle Datasets
3. Upload or copy notebook content
4. Enable GPU accelerator
5. Run cells

## Dataset Setup

### Option 1: Download PTB-XL

```bash
# Download from PhysioNet
wget https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip

# Extract
unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip -d data/

# Update notebook paths
# Change: /kaggle/input/ptb-xl-dataset/...
# To: ../data/ptb-xl/...
```

### Option 2: Use Kaggle Dataset

If running on Kaggle, the dataset is already available at:
```
/kaggle/input/ptb-xl-dataset/
```

### Option 3: Use Sample Data

For quick testing with sample data:
```python
# In the notebook, modify data loading to use samples
data_path = '../data/sample_data/'
```

## Key Notebook Sections

### 1. Data Preparation
- Load PTB-XL metadata
- Parse SCP diagnostic codes
- Create multilabel targets
- Train/validation/test split

### 2. Signal Processing
- Bandpass filtering (0.5-40 Hz)
- Notch filtering (50/60 Hz)
- Resampling to 100 Hz
- Z-score normalization

### 3. Model Architecture
- Parallel xLSTM implementation
- sLSTM and mLSTM branches
- Metadata integration
- Classification head

### 4. Training
- PyTorch Lightning trainer
- Early stopping callback
- Model checkpointing
- Learning rate scheduling

### 5. Evaluation
- Per-class metrics
- Confusion matrices
- ROC curves
- Precision-recall curves

### 6. Export
- Save model checkpoint
- Export normalization parameters
- Generate model summary

## Training Time

**Approximate training time** (full PTB-XL dataset):
- GPU (Tesla T4): ~2-3 hours
- GPU (V100): ~1-2 hours
- CPU: ~10-15 hours (not recommended)

**For quick testing** (10% of data):
- GPU: ~15-20 minutes
- CPU: ~1-2 hours

## Output Files

After training, the notebook generates:
- `xlstm_100hz_parallel_final.ckpt` - Model checkpoint
- `normalization_params.npz` - Mean/std for preprocessing
- `training_log.csv` - Training metrics history
- Various plots and visualizations

## Customization

### Modify Hyperparameters

Edit the CONFIG dictionary:
```python
CONFIG = {
    'embedding_dim': 128,        # Try: 64, 256
    'num_heads': 4,              # Try: 2, 8
    'dropout': 0.3,              # Try: 0.2, 0.4, 0.5
    'learning_rate': 0.001,      # Try: 0.0001, 0.01
    'batch_size': 32,            # Try: 16, 64
}
```

### Add Data Augmentation

```python
# Time shifting
shifted = np.roll(signal, shift=random.randint(-10, 10), axis=0)

# Amplitude scaling
scaled = signal * random.uniform(0.9, 1.1)

# Noise injection
noisy = signal + np.random.normal(0, 0.01, signal.shape)
```

### Experiment with Architecture

```python
# More sLSTM blocks
num_blocks=3  # Instead of 2

# Different hidden sizes
embedding_dim=256

# Additional dropout
dropout=0.4
```

## Troubleshooting

### Out of Memory
- Reduce batch size: `CONFIG['batch_size'] = 16`
- Use gradient accumulation
- Enable mixed precision training

### Slow Training
- Use GPU runtime
- Reduce model size
- Use smaller dataset subset for testing

### Poor Performance
- Check data preprocessing
- Verify label distribution
- Increase training epochs
- Adjust learning rate
- Add data augmentation

## Experiments to Try

1. **Different architectures**: CNN, Transformer, ResNet
2. **Ensemble methods**: Multiple models, voting
3. **Transfer learning**: Pre-train on other ECG datasets
4. **Attention visualization**: Identify important ECG segments
5. **Explainability**: Grad-CAM, SHAP values

## Additional Notebooks (Future)

- `data_exploration.ipynb` - EDA and visualization
- `preprocessing_experiments.ipynb` - Filter comparisons
- `model_comparison.ipynb` - Benchmark different architectures
- `attention_analysis.ipynb` - Interpret model attention

## Resources

- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/)
- [xLSTM Paper](https://arxiv.org/abs/2405.04517)
- [PTB-XL Paper](https://www.nature.com/articles/s41597-020-0495-6)
- [WFDB Tutorial](https://wfdb.readthedocs.io/)

## License

MIT License - See [LICENSE](../LICENSE) for details.
