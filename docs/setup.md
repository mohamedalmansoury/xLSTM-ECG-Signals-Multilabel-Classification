# Setup Guide

This guide will help you set up the xLSTM ECG Classification project on your local machine.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)
- (Optional) CUDA-compatible GPU for training

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/xLSTM-ECG-Signals-Multilabel-Classification.git
cd xLSTM-ECG-Signals-Multilabel-Classification
```

### 2. Create Virtual Environment

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

**For Training:**
```bash
pip install -r requirements.txt
```

**For Deployment Only:**
```bash
pip install -r deployment/requirements.txt
```

### 4. Download Model Checkpoint

The trained model is not included in the repository due to size. Download it from:

- **Google Drive**: [Link to be added]
- **HuggingFace**: [Link to be added]

Place the checkpoint file:
```bash
# For deployment
models/xlstm_100hz_parallel_final.ckpt

# Or in deployment folder
deployment/xlstm_100hz_parallel_final.ckpt
```

### 5. Download Dataset (Optional, for Training)

**PTB-XL Dataset:**

**Option A: PhysioNet**
```bash
wget https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
unzip ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip -d data/
```

**Option B: Kaggle**
```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (requires Kaggle API token)
kaggle datasets download -d khyeh0719/ptb-xl-dataset
unzip ptb-xl-dataset.zip -d data/
```

## Verification

### Test Installation

```python
# Test PyTorch
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test xLSTM
from xlstm import xLSTMBlockStack
print("xLSTM imported successfully")

# Test WFDB
import wfdb
print("WFDB imported successfully")
```

### Run Sample Prediction

```bash
cd deployment
streamlit run app.py
```

Then upload sample files from `data/sample_data/`.

## Troubleshooting

### Common Issues

#### Issue: ModuleNotFoundError for xlstm

**Solution:**
```bash
pip install xlstm
```

If that fails:
```bash
pip install git+https://github.com/NX-AI/xlstm.git
```

#### Issue: CUDA not available

**Solution:** For GPU support, install PyTorch with CUDA:
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Issue: WFDB DLL errors on Windows

**Solution:** Convert files to NumPy format:
```python
import wfdb
import numpy as np

signal, _ = wfdb.rdsamp('path/to/record')
np.save('signal.npy', signal)
```

#### Issue: Out of memory during training

**Solutions:**
1. Reduce batch size in config
2. Use gradient accumulation
3. Enable mixed precision training
4. Use smaller model variant

### Package Conflicts

If you encounter package conflicts:

```bash
# Create fresh environment
deactivate
rm -rf venv  # or Remove-Item venv -Recurse on Windows
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install packages one by one
pip install torch
pip install pytorch-lightning
pip install xlstm
pip install -r requirements.txt
```

## Platform-Specific Notes

### Windows

- Use PowerShell or Command Prompt
- Activate environment: `.\venv\Scripts\activate`
- May need to enable script execution: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### macOS

- May need to install Xcode Command Line Tools: `xcode-select --install`
- For M1/M2 Macs, use: `pip install torch torchvision`

### Linux

- May need to install system dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-dev python3-pip
  ```

## Development Setup

For contributing to the project:

```bash
# Install development dependencies
pip install -r requirements.txt
pip install jupyter black flake8 pytest

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## IDE Configuration

### VS Code

Recommended extensions:
- Python
- Jupyter
- Pylance
- GitLens

Settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black"
}
```

### PyCharm

1. File → Settings → Project → Python Interpreter
2. Add interpreter → Existing environment
3. Select `venv/bin/python` (or `venv\Scripts\python.exe` on Windows)

## Next Steps

After setup:

1. **Try the deployment app**: `streamlit run deployment/app.py`
2. **Explore the notebook**: `jupyter notebook notebooks/ECG_Signals_Classification_xLSTM.ipynb`
3. **Read the documentation**: Check `docs/` folder
4. **Run with sample data**: Use files in `data/sample_data/`

## Getting Help

- Read the [README.md](../README.md)
- Check [docs/architecture.md](architecture.md)
- Review [CONTRIBUTING.md](../CONTRIBUTING.md)
- Open an issue on GitHub

## License

MIT License - See [LICENSE](../LICENSE) for details.
