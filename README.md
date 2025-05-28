# 🧬 Quantum-CNN Synergy: Early Pancreatic Cancer Detection

This project leverages a hybrid **Quantum Convolutional Neural Network (QCNN)** model to detect pancreatic cancer at an early stage. It combines classical deep learning and quantum computing using **PyTorch** and **Pennylane**, based on biomarker data from the research dataset by *Debernardi et al., 2020*.

---

## 📌 Key Features

- 🔗 Hybrid Classical + Quantum model (QCNN)
- 📊 Uses 7 biomarker features for prediction
- 🧠 Implemented using PyTorch + Pennylane
- ✅ Trained and tested on real clinical data
- 📈 Accuracy: ~82% (optimized)

---

## 📁 Project Structure

.
├── pancreas_detection_optimized.pth # Trained hybrid model
├── scaler.pkl # Scaler object for preprocessing
├── training_script.py # Model training and saving
├── test_model.py # Script to evaluate on unseen test set
├── Debernardi et al 2020 data.csv # Clinical biomarker dataset
└── README.md # Project documentation


---

## 🧪 Dataset

- **Source**: Debernardi et al., 2020 (Nature Medicine)
- **Label**:
  - `diagnosis == 3` → Cancer
  - `diagnosis != 3` → Non-cancer
- **Features Used**:
  - `creatinine`
  - `plasma_CA19_9`
  - `age`
  - `sex` (M=1, F=0)
  - `LYVE1`
  - `REG1B`
  - `TFF1`

---

## ⚙️ How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/qcnn-pancreas-detection.git
   cd qcnn-pancreas-detection
