# Sample ECG Data

This folder contains sample ECG recordings for testing the classification model.

## Sample Cases

### 1. Normal Case
**File**: `00001_lr.hea` / `00001_lr.dat`
- **Patient**: 56-year-old male
- **Diagnosis**: Normal ECG (NORM)
- **Description**: No cardiac abnormalities detected

### 2. Hypertrophy Case
**File**: `00138_lr.hea` / `00138_lr.dat`
- **Patient**: 74-year-old male
- **Diagnosis**: Hypertrophy (HYP)
- **Description**: Thickening of heart muscle

### 3. Conduction Disturbance Case
**File**: `00157_lr.hea` / `00157_lr.dat`
- **Patient**: 74-year-old male
- **Diagnosis**: Conduction Disturbance (CD)
- **Description**: Abnormal electrical conduction

### 4. ST-T Change Case
**File**: `00292_lr.hea` / `00292_lr.dat`
- **Patient**: 83-year-old female
- **Diagnosis**: ST-T Change (STTC)
- **Description**: ST segment and T wave abnormalities

## Usage

### With Streamlit App

1. Start the deployment app:
```bash
cd deployment
streamlit run app.py
```

2. Upload both `.hea` and `.dat` files for a case
3. Enter the patient's age and sex
4. View the classification results

### Programmatically

```python
import wfdb
import numpy as np

# Load WFDB record
signal, fields = wfdb.rdsamp('data/sample_data/00001_lr')

# Signal is now a numpy array (samples, 12_leads)
print(f"Signal shape: {signal.shape}")
print(f"Sampling frequency: {fields['fs']} Hz")
print(f"Signal length: {fields['sig_len']} samples")
```

## File Format

These files are in **WFDB format** (WaveForm DataBase):
- **`.hea`**: Header file with metadata
- **`.dat`**: Binary data file with signal values

### Header File (.hea) Contents
- Record name
- Number of signals
- Sampling frequency
- Number of samples
- Signal specifications (gain, baseline, units)

### Data File (.dat) Contents
- Binary encoded ECG signal data
- 12 leads: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6

## Signal Characteristics

- **Sampling Frequency**: 100 Hz
- **Duration**: 10 seconds
- **Leads**: 12-lead ECG
- **Format**: WFDB binary

## Converting to Other Formats

### To NumPy
```python
import wfdb
import numpy as np

signal, _ = wfdb.rdsamp('00001_lr')
np.save('00001_lr.npy', signal)
```

### To CSV
```python
import wfdb
import numpy as np

signal, _ = wfdb.rdsamp('00001_lr')
np.savetxt('00001_lr.csv', signal, delimiter=',')
```

## Data Source

These samples are from the **PTB-XL Dataset**:
- Published by PhysioNet
- DOI: 10.13026/x4td-x982
- License: Open Database License (ODbL) v1.0

## Important Notes

⚠️ **Privacy**: These are publicly available research datasets. Patient identifiers have been removed.

⚠️ **Usage**: These samples are for testing and demonstration purposes only. Not for clinical use.

## References

Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F.I., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. Scientific Data, 7(1), 154.

## Need More Data?

Download the full PTB-XL dataset:
- **PhysioNet**: https://physionet.org/content/ptb-xl/
- **Kaggle**: https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset

## License

The sample data follows the original PTB-XL dataset license (ODbL v1.0).
