# 💤 Multimodal Sleep Stage Classification using Attention-Based Deep Learning

An end-to-end deep learning system for automated sleep stage classification using multimodal physiological signals (EEG, EOG, and EMG) from the PhysioNet Sleep-EDF dataset.

This project combines signal processing, multimodal learning, CNN-LSTM architectures, and attention mechanisms to classify sleep stages while improving generalization across unseen subjects.

---

## 📌 Overview

Sleep staging is a critical task in sleep analysis and diagnosis. Traditionally, sleep experts manually score sleep stages using polysomnography (PSG) recordings, which is time-consuming and expensive.

This project automates sleep stage classification by leveraging:

- EEG (Brain Activity)
- EOG (Eye Movement)
- EMG (Muscle Activity)

A multimodal attention-based deep learning architecture is used to learn complex relationships between these physiological signals and predict sleep stages automatically.

---

## 🎯 Objectives

- Build an end-to-end sleep stage classification pipeline.
- Process raw PSG recordings from Sleep-EDF.
- Extract meaningful features from EEG, EOG, and EMG signals.
- Learn temporal sleep dynamics using LSTMs.
- Improve multimodal feature fusion using attention mechanisms.
- Evaluate performance on unseen subjects.

---

# 📂 Dataset

### Sleep-EDF Expanded Dataset

Source:
https://physionet.org/content/sleep-edfx/

The dataset contains overnight polysomnography recordings along with expert-annotated sleep stages.

### Signals Used

| Signal | Description |
|----------|------------|
| EEG Fpz-Cz | Brain Activity |
| EOG Horizontal | Eye Movement |
| EMG Submental | Muscle Activity |

### Sleep Stages

| Label | Stage |
|---------|---------|
| W | Wake |
| N1 | Sleep Stage 1 |
| N2 | Sleep Stage 2 |
| N3/N4 | Deep Sleep |
| REM | Rapid Eye Movement |

---

# ⚙️ Data Processing Pipeline

## 1. Loading Raw EDF Files

Raw PSG and hypnogram files are loaded using the MNE library.

```python
raw = mne.io.read_raw_edf(...)
annotations = mne.read_annotations(...)
```

---

## 2. Signal Preprocessing

The following preprocessing steps are applied:

- Channel selection
- Bandpass filtering (0.3 – 35 Hz)
- Downsampling to 100 Hz
- Annotation synchronization
- Epoch generation

```python
raw.filter(0.3, 35)
raw.resample(100)
```

---

## 3. Epoch Creation

Signals are divided into standard 30-second epochs.

Each epoch becomes one training sample.

```text
EEG/EOG/EMG → 30-second Window → Sleep Stage Label
```

---

## 4. Normalization

Per-subject normalization is applied:

```text
(x - mean) / std
```

This reduces subject-specific bias and improves model generalization.

---

# 🏗 Model Architecture

The model follows a multimodal multi-branch architecture.

```text
           EEG
            │
        CNN + Attention
            │
           LSTM
            │

           EOG
            │
        CNN + Attention
            │
           LSTM
            │

           EMG
            │
        CNN + Attention
            │
           LSTM

             ↓

     Cross-Modal Attention

             ↓

      Feature Fusion

             ↓

     Global Pooling

             ↓

       Dense Layers

             ↓

     Sleep Stage Output
```

---

# 🧠 Feature Extraction

Each modality has its own specialist branch.

### CNN Layers

Used for:

- Frequency pattern extraction
- Local feature learning
- Noise reduction

### LSTM Layers

Used for:

- Temporal modeling
- Sleep transition learning
- Sequence dependency capture

---

# 🎯 Attention Mechanisms

## Channel Attention

Squeeze-and-Excitation blocks help the model focus on informative channels.

```text
Signal → Global Pooling → Importance Weights → Feature Recalibration
```

---

## Self Attention

Allows EEG features to strengthen important temporal patterns.

---

## Cross Attention

The model learns interactions between:

```text
EEG ↔ EOG
EEG ↔ EMG
```

This helps improve stage recognition by combining complementary physiological information.

---

# 🛡 Preventing Overfitting

Several techniques were used:

### Dropout

```python
Dropout(0.5)
```

### Batch Normalization

Improves training stability.

### Early Stopping

Stops training when validation loss no longer improves.

### ReduceLROnPlateau

Automatically lowers learning rate when progress stalls.

---

# 📊 Handling Data Leakage

A major challenge in biomedical machine learning is subject leakage.

Instead of random splitting, the project uses:

```python
GroupShuffleSplit
```

This ensures:

- Subjects in training set never appear in test set.
- Evaluation better reflects real-world performance.

---

# ⚖️ Handling Class Imbalance

Sleep datasets contain highly imbalanced classes.

To address this:

```python
class_weight.compute_class_weight(...)
```

was used to balance training.

---

# 📈 Evaluation Metrics

The model is evaluated using:

### Accuracy

Measures overall prediction correctness.

### Cohen's Kappa Score

Measures agreement while accounting for random chance.

### Classification Report

Provides:

- Precision
- Recall
- F1-score

for every sleep stage.

### Confusion Matrix

Visualizes class-wise prediction performance.

---

# 🛠 Tech Stack

## Machine Learning

- TensorFlow
- Keras
- Scikit-Learn

## Signal Processing

- MNE

## Data Processing

- NumPy
- Pandas

## Visualization

- Matplotlib
- Seaborn

## Environment

- Python
- Google Colab

---

# 📋 Project Workflow

```text
Sleep EDF Dataset
        ↓
Signal Preprocessing
        ↓
Epoch Generation
        ↓
Normalization
        ↓
Multimodal CNN-LSTM
        ↓
Attention Fusion
        ↓
Training
        ↓
Evaluation
        ↓
Sleep Stage Prediction
```

---

# 🚀 Future Improvements

### Transformer-Based Fusion

Replace attention blocks with multimodal transformers.

### Explainable AI

Integrate:

- SHAP
- Attention Visualizations
- Feature Attribution

### Real-Time Monitoring

Deploy the model as:

- Flask API
- FastAPI Service

for real-time sleep analysis.

### Larger Dataset Evaluation

Evaluate on:

- Sleep-EDF Expanded
- ISRUC Sleep Dataset
- MASS Sleep Dataset

---

# 🎓 Key Learnings

Through this project, I gained hands-on experience in:

- EEG/EOG/EMG Analysis
- Deep Learning for Time-Series Data
- Attention Mechanisms
- Multimodal Learning
- Model Generalization
- Handling Data Leakage
- Sleep Analytics

---
