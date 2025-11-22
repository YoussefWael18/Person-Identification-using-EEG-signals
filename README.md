# EEG-Based Person Identification using Deep Learning 

A deep learning system for person identification using EEG (electroencephalography) signals, achieving biometric authentication through unique "brainprints" extracted from brain activity patterns.

## Project Overview

This project implements a complete pipeline for identifying individuals based on their unique EEG signal patterns. Unlike traditional motor imagery classification, this system focuses on **person identification** rather than task classification, treating each person's brain activity as a unique biometric signature.

### Key Features

- **109 Subject Classification**: Identifies individuals from a dataset of 109 subjects
- **Robust Preprocessing Pipeline**: Comprehensive signal processing including filtering, re-referencing, and artifact removal
- **Feature Engineering**: Extracts 1,280 features per epoch (20 features × 64 channels) combining time-domain and frequency-domain characteristics
- **Deep Neural Network**: Multi-layer architecture with batch normalization and dropout for robust classification
- **High Accuracy**: Achieves strong performance in person identification tasks
- **Complete Workflow**: From raw EEG data to trained model with visualization tools

## Project Structure

```
EEG/
├── 1.0.0/                          # Raw EEG dataset (109 subjects)
│   ├── S001/ ... S109/             # Individual subject folders
│   ├── epochs_data/                # Processed 2-second epochs
│   ├── features_data/              # Extracted features per subject
│   └── deep_learning_data/         # Normalized data ready for training
├── models/                         # Trained models and visualizations
│   ├── best_model.keras            # Best performing model
│   ├── eeg_person_identification_model.keras
│   └── *.png                       # Training and visualization plots
├── preprocessing.ipynb             # Data preprocessing pipeline
├── EEG_Visualization.ipynb         # Exploratory data analysis
└── Deep_learing.ipynb              # Model training and evaluation
```

## 🔬 Methodology

### 1. Data Preparation

**Dataset**: PhysioNet EEG Motor Movement/Imagery Dataset
- **Subjects**: 109 individuals
- **Channels**: 64 EEG electrodes
- **Sampling Rate**: Standardized to 128 Hz
- **Recording Sessions**: 14 runs per subject (R01-R14)

**Preprocessing Steps**:
1. **File Concatenation**: Combined 14 runs per subject into single continuous recordings
2. **Resampling**: Standardized all recordings to 128 Hz (some subjects recorded at 160 Hz)
3. **Band-pass Filtering**: 1-40 Hz to retain relevant brain activity
4. **Notch Filtering**: Removed 50 Hz power-line interference
5. **Re-referencing**: Applied Common Average Reference (CAR) to reduce noise

### 2. Epoching & Feature Extraction

**Epoching**:
- **Duration**: 2-second windows
- **Overlap**: 50% (1-second stride) for data augmentation
- **Result**: ~2,000-3,000 epochs per subject

**Feature Extraction** (20 features per channel × 64 channels = 1,280 features):

**Time-Domain Features** (10):
- Mean, Standard Deviation, Median
- Min, Max, Peak-to-Peak
- Mean Absolute Value
- Skewness, Kurtosis, RMS

**Frequency-Domain Features** (10):
- Band Power: Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-40 Hz)
- PSD Statistics: Mean, Std, Median, Dominant Frequency, Max PSD

### 3. Deep Learning Model

**Architecture**:
```
Input Layer (1,280 features)
    ↓
Dense (512 neurons) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense (256 neurons) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense (128 neurons) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Output Layer (109 neurons, Softmax)
```

**Training Configuration**:
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Regularization**: Batch Normalization + Dropout
- **Data Split**: 80% training, 20% testing (stratified)

##  Getting Started

### Prerequisites

```bash
pip install numpy scipy scikit-learn
pip install mne matplotlib seaborn pandas
pip install tensorflow keras
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/EEG-Person-Identification.git
cd EEG-Person-Identification
```

2. **Download the dataset**:
   - Download the PhysioNet EEG Motor Movement/Imagery Dataset
   - Extract to `1.0.0/` directory

3. **Run the preprocessing pipeline**:
```bash
jupyter notebook preprocessing.ipynb
```

4. **Train the model**:
```bash
jupyter notebook Deep_learing.ipynb
```

### Usage

**Quick Start - Using Pre-trained Model**:

```python
import numpy as np
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('models/best_model.keras')

# Load test data
X_test = np.load('1.0.0/deep_learning_data/X_normalized.npy')
y_test = np.load('1.0.0/deep_learning_data/y_encoded.npy')

# Make predictions
predictions = model.predict(X_test)
predicted_person = np.argmax(predictions, axis=1)
```

**Training from Scratch**:

Follow the notebooks in order:
1. `preprocessing.ipynb` - Preprocess raw EEG data
2. `EEG_Visualization.ipynb` - Explore the processed data
3. `Deep_learing.ipynb` - Train and evaluate the model

## Results

The model demonstrates strong performance in identifying individuals based on their EEG patterns:

- **Training Accuracy**: High accuracy on training set
- **Test Accuracy**: Robust generalization to unseen data
- **Feature Importance**: Frequency-domain features (especially Alpha and Beta bands) show strong discriminative power

### Visualizations

The project includes comprehensive visualizations:
- Sample distribution across subjects
- EEG signal patterns for different individuals
- Feature value distributions
- Training history (loss and accuracy curves)

##  Key Insights

1. **Unique Brainprints**: Each person exhibits distinct EEG patterns that remain consistent across different tasks
2. **Frequency Bands**: Alpha (8-13 Hz) and Beta (13-30 Hz) bands are particularly discriminative
3. **Channel Importance**: Different electrode locations contribute varying levels of discriminative information
4. **Temporal Stability**: 2-second epochs provide sufficient information for reliable identification

## Technical Details

### Why Person Identification vs Task Classification?

Traditional EEG analysis focuses on classifying tasks (e.g., "left hand movement" vs "right hand movement"). This project takes a different approach:

- **Label = Person ID** (not task type)
- **Goal**: Create a biometric authentication system
- **Application**: Secure authentication, forensics, medical diagnostics

### Data Processing Rationale

**Why ignore `.event` files?**
- Event files mark task boundaries (left/right hand imagery, rest periods)
- For person identification, we don't care *what* they were doing
- We only care *who* they are

**Why concatenate all runs?**
- Creates a robust signature across various mental states
- Increases data per subject for better model training
- Captures person-specific patterns independent of task

## Future Improvements

- [ ] Implement CNN-based architecture for spatial feature learning
- [ ] Add LSTM layers for temporal pattern recognition
- [ ] Explore transfer learning from pre-trained EEG models
- [ ] Real-time person identification system
- [ ] Cross-dataset validation
- [ ] Attention mechanisms for channel importance

##  References

- **Dataset**: Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004). BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. IEEE TBME 51(6):1034-1043
- **PhysioNet**: Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation 101(23):e215-e220

## License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

## Acknowledgments

- PhysioNet for providing the EEG dataset
- MNE-Python community for excellent EEG processing tools
- TensorFlow/Keras team for the deep learning framework

---

**Note**: This is a research/educational project. For production biometric systems, additional security measures and validation would be required.
