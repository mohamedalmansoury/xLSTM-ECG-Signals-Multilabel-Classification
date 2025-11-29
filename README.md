# xLSTM ECG Signals Multilabel Classification

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

🚀 **[Try Live Demo](https://xlstm-ecg-signals-multilabel-classification-almansoury.streamlit.app/)** | 📂 **[GitHub](https://github.com/mohamedalmansoury/xLSTM-ECG-Signals-Multilabel-Classification)**

A deep learning project for multilabel classification of ECG signals using Parallel xLSTM architecture. This project classifies ECG recordings into five cardiac conditions: Normal (NORM), Myocardial Infarction (MI), ST-T Change (STTC), Conduction Disturbance (CD), and Hypertrophy (HYP).

## 🔍 Project Overview

This project implements a novel parallel xLSTM architecture combining sLSTM (spatial) and mLSTM (memory) blocks for accurate multilabel classification of 12-lead ECG signals. The model is trained on the PTB-XL dataset and deployed as an interactive Streamlit web application.

### Key Features

- **Parallel xLSTM Architecture**: Combines sLSTM and mLSTM blocks for enhanced temporal pattern recognition
- **Multilabel Classification**: Predicts multiple cardiac conditions simultaneously
- **Signal Processing Pipeline**: Advanced ECG preprocessing with bandpass and notch filters
- **Interactive Web App**: User-friendly Streamlit deployment for real-time predictions
- **Multiple Input Formats**: Supports WFDB (.dat/.hea), NumPy (.npy), and CSV formats
- **Patient Metadata Integration**: Incorporates age and sex for improved predictions

## 📊 Dataset

The model is trained on the **PTB-XL Dataset**, a large publicly available electrocardiography dataset:
- **21,837 clinical 12-lead ECG recordings**
- **10 seconds duration per recording**
- **500 Hz sampling rate** (downsampled to 100 Hz)
- **Five superclass labels**: NORM, MI, STTC, CD, HYP

Dataset source: [PhysioNet PTB-XL](https://physionet.org/content/ptb-xl/)

## 🏗️ Model Architecture

### Parallel xLSTM Components

1. **Signal Embedding Layer**
   - Linear projection to 128-dimensional space
   - Layer normalization + ReLU activation

2. **Parallel Processing Branches**
   - **sLSTM Branch**: 2 blocks with 4 attention heads for spatial patterns
   - **mLSTM Branch**: 2 blocks with 4 attention heads for temporal memory

3. **Feature Fusion**
   - Average pooling of parallel branches
   - Global average + max pooling

4. **Metadata Integration**
   - Separate network for patient age and sex
   - Concatenation with ECG features

5. **Classification Head**
   - Three-layer MLP with dropout
   - Sigmoid activation for multilabel output

### Model Configuration

```python
CONFIG = {
    'input_shape': (1000, 12),      # 10s @ 100Hz, 12 leads
    'num_classes': 5,                # NORM, MI, STTC, CD, HYP
    'metadata_dim': 2,               # Age, Sex
    'embedding_dim': 128,
    'slstm_hidden_size': 256,
    'mlstm_hidden_size': 256,
    'num_heads': 4,
    'dropout': 0.3,
    'learning_rate': 0.001,
}
```

## 🌐 Live Demo

**Try it now:** [https://xlstm-ecg-signals-multilabel-classification-almansoury.streamlit.app/](https://xlstm-ecg-signals-multilabel-classification-almansoury.streamlit.app/)

Upload ECG files and get instant predictions without any setup!

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/xLSTM-ECG-Signals-Multilabel-Classification.git
cd xLSTM-ECG-Signals-Multilabel-Classification
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

For training:
```bash
pip install -r requirements.txt
```

For deployment only:
```bash
pip install -r deployment/requirements.txt
```

## 📖 Usage

### Training the Model

1. **Download PTB-XL Dataset**
```bash
# Download from PhysioNet or Kaggle
wget https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1.zip
```

2. **Run Training Notebook**
```bash
jupyter notebook notebooks/ECG_Signals_Classification_xLSTM.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Signal filtering and normalization
- Model training with PyTorch Lightning
- Evaluation metrics and visualization
- Model checkpoint saving

### Running the Deployment App

1. **Place model checkpoint**
```bash
# Copy or download the trained model
cp path/to/xlstm_100hz_parallel_final.ckpt models/
```

2. **Start the Streamlit app**
```bash
cd deployment
streamlit run app.py
```

3. **Open your browser** to `http://localhost:8501`

4. **Upload ECG files** in supported formats:
   - WFDB format (.dat + .hea files together)
   - NumPy array (.npy)
   - CSV file (.csv)

5. **Enter patient metadata** (age, sex) in the sidebar

6. **View predictions** with confidence scores and interpretations

### Using Sample Data

Try the app with provided sample ECG recordings:
```bash
cd deployment
streamlit run app.py
# Upload files from ../data/sample_data/
```

Sample cases included:
- Normal (56-year-old male)
- Hypertrophy (74-year-old male)
- Conduction Disturbance (74-year-old male)
- ST-T Change (83-year-old female)

## 📁 Project Structure

```
xLSTM-ECG-Signals-Multilabel-Classification/
│
├── notebooks/                              # Jupyter notebooks
│   └── ECG_Signals_Classification_xLSTM.ipynb
│
├── deployment/                             # Streamlit web app
│   ├── app.py                             # Main application
│   ├── model_inference.py                 # Model loading & inference
│   ├── utils.py                           # Signal processing utilities
│   ├── requirements.txt                   # Deployment dependencies
│   └── README.md                          # Deployment guide
│
├── src/                                    # Source code modules
│   ├── __init__.py
│   ├── models.py                          # Model architecture
│   ├── data_loader.py                     # Data loading utilities
│   └── preprocessing.py                   # Signal preprocessing
│
├── data/                                   # Data directory
│   ├── sample_data/                       # Sample ECG files
│   └── README.md                          # Data documentation
│
├── models/                                 # Trained model checkpoints
│   ├── xlstm_100hz_parallel_final.ckpt   # Main model checkpoint
│   └── normalization_params.npz           # Normalization parameters
│
├── docs/                                   # Additional documentation
│   ├── architecture.md                    # Detailed architecture
│   ├── training.md                        # Training guide
│   └── api.md                             # API documentation
│
├── requirements.txt                        # Training dependencies
├── .gitignore                             # Git ignore rules
├── LICENSE                                # MIT License
└── README.md                              # This file
```

## 🔬 Signal Processing

### Preprocessing Pipeline

1. **Bandpass Filtering**
   - High-pass: 0.5 Hz (removes baseline wander)
   - Low-pass: 40 Hz (removes high-frequency noise)
   - Butterworth filter, order 3

2. **Notch Filtering**
   - 50 Hz (60 Hz for US) power line interference removal
   - Quality factor: 30

3. **Normalization**
   - Z-score normalization using training set statistics
   - Per-lead normalization

4. **Resampling**
   - Original: 500 Hz
   - Target: 100 Hz (for computational efficiency)
   - 1000 samples = 10 seconds

## 📈 Performance Metrics

The model is evaluated using:
- **F1 Score** (macro and per-class)
- **ROC-AUC** (macro and per-class)
- **Precision & Recall**
- **Accuracy**
- **Confusion Matrix**

Typical performance on PTB-XL test set:
- Macro F1 Score: ~0.85
- Macro ROC-AUC: ~0.92
- Per-class performance varies by condition prevalence

## 🛠️ Technologies Used

- **Deep Learning**: PyTorch, PyTorch Lightning, xLSTM
- **Signal Processing**: SciPy, WFDB
- **Web Framework**: Streamlit
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Data Processing**: NumPy, Pandas
- **Model Evaluation**: Scikit-learn

## 📝 Clinical Labels

| Abbr. | Full Name | Description |
|-------|-----------|-------------|
| NORM | Normal | No cardiac abnormalities detected |
| MI | Myocardial Infarction | Heart attack (acute or old) |
| STTC | ST-T Change | ST segment and T wave abnormalities |
| CD | Conduction Disturbance | Abnormal electrical conduction (bundle branch blocks, etc.) |
| HYP | Hypertrophy | Cardiac muscle thickening (left or right ventricle/atrium) |

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PTB-XL Dataset**: PhysioNet and the dataset authors
- **xLSTM Library**: Maximilian Beck et al. for the xLSTM implementation
- **PyTorch Lightning**: For the excellent training framework
- **Streamlit**: For the easy-to-use web app framework

## 📚 References

1. **PTB-XL Dataset**:
   - Wagner et al. (2020). "PTB-XL, a large publicly available electrocardiography dataset"
   
2. **xLSTM Architecture**:
   - Beck et al. (2024). "xLSTM: Extended Long Short-Term Memory"

3. **ECG Classification**:
   - Ribeiro et al. (2020). "Automatic diagnosis of the 12-lead ECG using a deep neural network"

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for research and educational purposes. Always consult healthcare professionals for medical diagnosis.
