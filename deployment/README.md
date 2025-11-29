# ECG Deployment Application

🌐 **Live Demo:** [https://xlstm-ecg-signals-multilabel-classification-almansoury.streamlit.app/](https://xlstm-ecg-signals-multilabel-classification-almansoury.streamlit.app/)

This folder contains the Streamlit web application for deploying the xLSTM ECG classification model.

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Ensure model files are present**:
   - `xlstm_100hz_parallel_final.ckpt` (model checkpoint)
   - `normalization_params.npz` (preprocessing parameters)

3. **Run the app**:
```bash
streamlit run app.py
```

4. **Access the app**: Open your browser to `http://localhost:8501`

## File Descriptions

- **`app.py`**: Main Streamlit application interface
- **`model_inference.py`**: Model loading and inference logic
- **`utils.py`**: Signal preprocessing utilities (filtering, normalization)
- **`requirements.txt`**: Python dependencies for deployment
- **`xlstm_100hz_parallel_final.ckpt`**: Trained model checkpoint
- **`normalization_params.npz`**: Global mean/std for signal normalization

## Supported Input Formats

### WFDB Format (.dat + .hea)
Upload both `.dat` and `.hea` files together:
```
00001_lr.dat
00001_lr.hea
```

### NumPy Array (.npy)
Signal shape: `(1000, 12)` or `(12, 1000)`
- 1000 time steps (10 seconds @ 100 Hz)
- 12 ECG leads

### CSV Format (.csv)
Comma-separated values with shape `(1000, 12)`

## Usage Instructions

1. **Upload ECG File(s)**:
   - Drag and drop or browse for files
   - For WFDB format, upload both .dat and .hea files

2. **Enter Patient Metadata**:
   - Age (0-120 years)
   - Sex (Male/Female)

3. **View Predictions**:
   - Primary prediction with confidence
   - Probability distribution for all classes
   - Clinical interpretation

## Model Output

The model predicts probabilities for 5 cardiac conditions:
- **NORM**: Normal ECG
- **MI**: Myocardial Infarction
- **STTC**: ST-T Change
- **CD**: Conduction Disturbance
- **HYP**: Hypertrophy

## Signal Processing Pipeline

1. **Bandpass Filtering**: 0.5-40 Hz
2. **Notch Filtering**: 50/60 Hz power line removal
3. **Normalization**: Z-score using training set statistics
4. **Resampling**: To 100 Hz if needed
5. **Padding/Truncation**: To 1000 samples (10 seconds)

## Troubleshooting

### Model Not Found
Ensure `xlstm_100hz_parallel_final.ckpt` is in the deployment folder.

### WFDB Loading Issues
On Windows, WFDB may have DLL issues. Alternative: save as `.npy` format:
```python
import numpy as np
import wfdb

# Load WFDB record
signal, _ = wfdb.rdsamp('path/to/record')

# Save as NumPy
np.save('ecg_signal.npy', signal)
```

### Shape Mismatch
Verify signal shape is `(1000, 12)`:
- 1000 samples = 10 seconds @ 100 Hz
- 12 leads = I, II, III, aVR, aVL, aVF, V1-V6

### Low Confidence Predictions
- Check signal quality
- Verify correct lead placement
- Ensure proper preprocessing
- Confirm patient metadata is accurate

## Performance Notes

- **CPU-only deployment**: Model uses vanilla backends for CPU inference
- **First prediction**: May take a few seconds for model loading
- **Subsequent predictions**: Fast (<1 second)

## API Integration

To integrate the model into your own application:

```python
from model_inference import load_model_from_checkpoint
from utils import preprocess_input, load_normalization_params
import torch
import numpy as np

# Configuration
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
model = load_model_from_checkpoint('xlstm_100hz_parallel_final.ckpt', CONFIG)
global_mean, global_std = load_normalization_params('normalization_params.npz')

# Prepare input
signal = ...  # Your ECG signal (1000, 12)
metadata = [age, sex]  # [60, 1] for 60-year-old male
X, M = preprocess_input(signal, metadata, global_mean, global_std)

# Inference
X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
M = torch.tensor(M, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    output = model(X, M)
    probs = output.cpu().numpy()[0]

# Classes: ['NORM', 'MI', 'STTC', 'CD', 'HYP']
print(f"Predictions: {probs}")
```

## Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Docker (Future)
```bash
docker build -t ecg-classifier .
docker run -p 8501:8501 ecg-classifier
```

### Cloud Deployment
- **Streamlit Cloud**: Deploy directly from GitHub
- **Heroku**: Use Procfile with streamlit server
- **AWS/GCP/Azure**: Deploy as containerized service

## Security Notes

⚠️ **Important**: 
- This is a research tool, not a medical device
- Do not use for clinical diagnosis
- Patient data should be handled according to HIPAA/GDPR
- Consider adding authentication for production use

## Support

For issues or questions about deployment:
1. Check the main [README.md](../README.md)
2. Review [CONTRIBUTING.md](../CONTRIBUTING.md)
3. Open an issue on GitHub

## License

MIT License - See [LICENSE](../LICENSE) for details.
