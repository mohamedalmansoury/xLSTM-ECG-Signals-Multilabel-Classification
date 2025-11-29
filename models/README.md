# Models Directory

This directory stores trained model checkpoints and related artifacts.

## Files

### Model Checkpoint
- **`xlstm_100hz_parallel_final.ckpt`** (⚠️ Not included in repository)
  - Trained xLSTM model weights
  - Size: ~80-120 MB
  - Download link: [Add your model hosting link]

### Normalization Parameters
- **`normalization_params.npz`**
  - Global mean and standard deviation
  - Used for signal preprocessing
  - Size: <1 KB

## Downloading the Model

The trained model checkpoint is too large for GitHub. Download it from:

**Option 1: Google Drive**
```bash
# Download manually or use gdown
pip install gdown
gdown <google-drive-file-id> -O models/xlstm_100hz_parallel_final.ckpt
```

**Option 2: Hugging Face Hub**
```bash
# If hosted on Hugging Face
huggingface-cli download <repo-id> xlstm_100hz_parallel_final.ckpt --local-dir models/
```

**Option 3: Train Your Own**
```bash
# Train from scratch using the notebook
jupyter notebook notebooks/ECG_Signals_Classification_xLSTM.ipynb
```

## Model Information

### Architecture
- **Type**: Parallel xLSTM
- **Input**: (1000, 12) ECG signal + (2,) metadata
- **Output**: (5,) probabilities for [NORM, MI, STTC, CD, HYP]
- **Parameters**: ~15-20M trainable parameters

### Training Details
- **Dataset**: PTB-XL (21,837 ECG recordings)
- **Epochs**: 14 (with early stopping)
- **Batch Size**: 32
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Binary Cross-Entropy

### Performance
- **Macro F1 Score**: ~0.85
- **Macro ROC-AUC**: ~0.92
- **Validation Accuracy**: ~88%

## Using the Model

### In Python

```python
from deployment.model_inference import load_model_from_checkpoint
import torch

CONFIG = {
    'input_shape': (1000, 12),
    'num_classes': 5,
    'metadata_dim': 2,
    'embedding_dim': 128,
    'slstm_hidden_size': 256,
    'mlstm_hidden_size': 256,
    'num_heads': 4,
    'dropout': 0.3,
}

# Load model
model = load_model_from_checkpoint(
    'models/xlstm_100hz_parallel_final.ckpt',
    CONFIG
)

# Inference
model.eval()
with torch.no_grad():
    output = model(signal_tensor, metadata_tensor)
```

### With Streamlit App

```bash
cd deployment
streamlit run app.py
```

The app automatically loads the model from `deployment/xlstm_100hz_parallel_final.ckpt`.

## Model Versioning

### Current Version: v1.0
- Initial release
- Trained on PTB-XL dataset
- Parallel xLSTM architecture

### Future Versions
- v1.1: Fine-tuned with additional data
- v2.0: Enhanced architecture
- v2.1: Compressed model for edge devices

## Model Card

### Intended Use
- **Primary**: Research and educational purposes
- **Secondary**: Demonstration of xLSTM for time series classification
- **NOT intended**: Clinical diagnosis or medical decision-making

### Limitations
- Trained on specific patient population (PTB-XL)
- May not generalize to different ECG devices
- Performance varies by condition prevalence
- Requires proper ECG preprocessing

### Ethical Considerations
- Model predictions should not replace medical professionals
- Potential biases from training data demographics
- Privacy concerns with patient data
- Requires validation before any clinical use

### Training Data
- **Source**: PTB-XL dataset (PhysioNet)
- **Size**: 21,837 10-second 12-lead ECGs
- **Demographics**: Various ages, both sexes
- **Labels**: 5 superclasses (NORM, MI, STTC, CD, HYP)

### Evaluation Data
- **Test Set**: 20% of PTB-XL (stratified split)
- **Metrics**: F1, ROC-AUC, Precision, Recall, Accuracy

## Checkpoint Format

The checkpoint file contains:
```python
{
    'state_dict': {...},           # Model weights
    'hyper_parameters': {...},     # Training config
    'epoch': 14,                   # Training epoch
    'global_step': ...,            # Training step
    'pytorch-lightning_version': '2.x.x'
}
```

## Converting to Other Formats

### ONNX Export
```python
import torch.onnx

dummy_signal = torch.randn(1, 1000, 12)
dummy_metadata = torch.randn(1, 2)

torch.onnx.export(
    model,
    (dummy_signal, dummy_metadata),
    'models/xlstm_ecg.onnx',
    input_names=['signal', 'metadata'],
    output_names=['probabilities'],
    dynamic_axes={
        'signal': {0: 'batch_size'},
        'metadata': {0: 'batch_size'},
        'probabilities': {0: 'batch_size'}
    }
)
```

### TorchScript
```python
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save('models/xlstm_ecg.pt')
```

## Storage Recommendations

- Keep only the final/best model in version control
- Use Git LFS for model files >100MB
- Consider model compression techniques
- Host large models externally (HuggingFace, S3, etc.)

## Backup

Recommended backup locations:
1. Cloud storage (Google Drive, Dropbox)
2. Model registry (HuggingFace Hub, MLflow)
3. Institutional storage
4. Multiple local copies

## License

Model weights are released under MIT License - See [LICENSE](../LICENSE)

Dataset (PTB-XL) is under Open Database License (ODbL) v1.0

## Citation

If you use this model in your research, please cite:

```bibtex
@software{xlstm_ecg_classifier,
  author = {Your Name},
  title = {xLSTM ECG Signals Multilabel Classification},
  year = {2025},
  url = {https://github.com/yourusername/xLSTM-ECG-Signals-Multilabel-Classification}
}
```

## Contact

For questions about the model:
- Open an issue on GitHub
- Check the main [README.md](../README.md)
- Review [docs/architecture.md](../docs/architecture.md)
