
# MNIST Digit Classifier using Deep Neural Network (PyTorch)

This project implements a deep fully connected neural network to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) using PyTorch. The network is trained and evaluated on CPU, making it runnable even without GPU support.

---

## 🧠 Model Architecture

- Input Layer: 784 units (28×28 flattened image)
- Hidden Layer 1: 256 units + ReLU
- Hidden Layer 2: 128 units + ReLU
- Output Layer: 10 units (for digits 0 to 9)

---

## 📁 Project Structure

```
.
├── data/                   # MNIST data will be stored here
├── images/                 # Sample digit visualization (saved after training)
├── LICENSE.txt             # License information
├── README.md               # Project documentation
├── main.py                 # Main training, evaluation, and visualization code
└── requirements.txt        # Python dependencies
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/mnist-classifier.git
cd mnist-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the model
```bash
python main.py
```
---

## 📊 Training Output

The model trains over 10 epochs and displays loss at each epoch:

```
Epoch 1, Loss: 260.6516  
Epoch 2, Loss: 99.4128  
Epoch 3, Loss: 65.2298  
Epoch 4, Loss: 49.0885  
Epoch 5, Loss: 37.8544  
Epoch 6, Loss: 28.4589  
Epoch 7, Loss: 22.5545  
Epoch 8, Loss: 19.3613  
Epoch 9, Loss: 16.6477  
Epoch 10, Loss: 14.9400  
```

✅ **Final Accuracy**:  
- **Training Accuracy**: 99.78%  
- **Test Accuracy**: 97.99%

📌 **Sample Prediction Output**:  
- **Predicted Label**: `1` (for a randomly selected digit)

---

## 🔧 Requirements

- Python ≥ 3.6  
- PyTorch  
- torchvision  
- matplotlib  

You can install them via:
```bash
pip install torch torchvision matplotlib
```

---

## 📌 Notes

- The model runs entirely on **CPU**.
- You can extend this to use **CNNs**, **GPU support**, or **hyperparameter tuning** for better performance.

---

## 🖼️ Sample Output

A randomly selected MNIST digit with its prediction saved as `sample_photo.png`:

![Sample Output](images/sample_photo.png)

---

## 📄 License

This project is licensed under the terms of the MIT license. See `LICENSE.txt` for details.
