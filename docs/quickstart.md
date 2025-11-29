# Quick Start Guide

Get started with xLSTM ECG Classification in 5 minutes!

## 🌐 Fastest Way: Use Live Demo

**No installation needed!** Try the app instantly:

👉 **[https://xlstm-ecg-signals-multilabel-classification-almansoury.streamlit.app/](https://xlstm-ecg-signals-multilabel-classification-almansoury.streamlit.app/)**

1. Open the link
2. Upload ECG files from `data/sample_data/`
3. Enter patient age and sex
4. Get instant predictions!

---

## 🎯 Quick Examples

### Example 1: Predict from Python
```python
import torch
import numpy as np
from deployment.model_inference import load_model_from_checkpoint
from deployment.utils import preprocess_input, load_normalization_params

# Load model
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

model = load_model_from_checkpoint('deployment/xlstm_100hz_parallel_final.ckpt', CONFIG)
mean, std = load_normalization_params('deployment/normalization_params.npz')

# Prepare input
signal = np.random.randn(1000, 12).astype(np.float32)  # Your ECG signal
metadata = [60, 1]  # 60-year-old male

X, M = preprocess_input(signal, metadata, mean, std)
X = torch.tensor(X).unsqueeze(0)
M = torch.tensor(M).unsqueeze(0)

# Predict
with torch.no_grad():
    probs = model(X, M).numpy()[0]

classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
for cls, prob in zip(classes, probs):
    print(f"{cls}: {prob:.4f}")
```

### Example 2: Load WFDB File
```python
import wfdb
import numpy as np

# Load ECG record
signal, fields = wfdb.rdsamp('data/sample_data/00001_lr')
print(f"Signal shape: {signal.shape}")
print(f"Sampling rate: {fields['fs']} Hz")

# Save as NumPy (easier to use)
np.save('ecg_signal.npy', signal)
```

### Example 3: Custom Preprocessing
```python
from src.preprocessing import clean_ecg_signal, normalize_signal

# Your raw ECG signal (1000, 12)
raw_signal = np.random.randn(1000, 12)

# Clean and normalize
clean = clean_ecg_signal(raw_signal)
normalized, mean, std = normalize_signal(clean)

print(f"Normalized shape: {normalized.shape}")
```

---

## 📊 Sample Output

When you run the app with sample data, you'll see:

```
🎯 Prediction: NORM
   Confidence: 95.2%

📈 All Probabilities:
   NORM: 0.9520
   MI:   0.0123
   STTC: 0.0234
   CD:   0.0089
   HYP:  0.0156
```

---

## 🔧 Troubleshooting

### Problem: "Model not found"
**Solution:** Ensure `xlstm_100hz_parallel_final.ckpt` is in the deployment folder.

### Problem: "xlstm not installed"
**Solution:** 
```bash
pip install xlstm
```

### Problem: "Out of memory"
**Solution:** Reduce batch size in training config or use CPU for inference.

### Problem: WFDB errors on Windows
**Solution:** Convert to .npy format:
```python
import wfdb
import numpy as np
signal, _ = wfdb.rdsamp('record_name')
np.save('signal.npy', signal)
```

---

## 📚 Learn More

- **Architecture**: See [docs/architecture.md](../docs/architecture.md)
- **Full Setup**: See [docs/setup.md](../docs/setup.md)
- **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Main README**: See [README.md](../README.md)

---

## 🎓 What You Can Do Next

1. **Try different ECG samples**: Test various cardiac conditions
2. **Modify the model**: Experiment with hyperparameters in the notebook
3. **Add visualizations**: Enhance the Streamlit app with custom plots
4. **Deploy to cloud**: Try Streamlit Cloud, Heroku, or AWS
5. **Contribute**: Add features, fix bugs, improve docs

---

## 💡 Tips

- Start with sample data to verify everything works
- Use GPU for training (much faster)
- CPU is fine for deployment/inference
- Check the notebook for detailed explanations
- Read error messages carefully - they usually point to the solution

---

## ⚠️ Important Notes

- This is a **research tool**, not for clinical use
- Always validate predictions with medical professionals
- Respect patient privacy and data regulations
- The model works best with proper ECG preprocessing

---

## 🤝 Need Help?

- Open an issue on GitHub
- Check existing issues for solutions
- Read the full documentation
- Review the code comments

---

**Happy Classifying! 🎉**
