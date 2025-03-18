# 🚀 Federated Learning with FLWR - Optimized Client
### Author - Aryan Pandey
## 📌 Overview
This project implements a **customized federated learning client** using [Flower (FLWR)](https://flower.ai/) with **client-side optimizations** to make training efficient on **constrained, embedded, and edge devices** (≤2GB RAM).

### 🔹 **Key Optimizations**
✅ **Model Optimization** → Quantization & Pruning for memory-efficient training.  
✅ **Partial Local Training** → Train only selected layers to reduce computation.  
✅ **Gradient Compression** → Reduces communication overhead during federated training.  
✅ **Asynchronous & Selective Training** → Optimizes training cycles based on device status.  
✅ **Secure Updates** → Implements Differential Privacy for data security.  

---
## ⚙️ Installation
### **1️⃣ Prerequisites**
Ensure you have the following installed:
- Python 3.8+
- PyTorch  
- Flower (FLWR)  
- MLflow for experiment tracking  

### **2️⃣ Clone the Repository**
```bash
git clone https://github.com/ltd-ARYAN-pvt/federated-learning-client.git
cd federated-learning-client
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---
## 🚀 Running the Project
### **1️⃣ Start the FLWR Server and CLients**
```bash
python train.py --disable-warnings
```


---
## 🔥 Features & Optimizations
### **1️⃣ Model Optimization**  
🔹 **Quantization** → Converts 32-bit weights to 8-bit to reduce memory usage.  
🔹 **Pruning** → Removes unnecessary neurons to reduce computations.  

### **2️⃣ Partial Local Training**  
🔹 Only fine-tunes the **last few layers**, reducing RAM & CPU usage.  

### **3️⃣ Efficient Data Transmission**  
🔹 **Gradient Compression** → Sends only important updates to the FL server.  

### **4️⃣ Asynchronous & Selective Training**  
🔹 **Trains only when needed** (e.g., when connected to WiFi or power).  

### **5️⃣ Secure Client Updates**  
🔹 Implements **Differential Privacy (DP)** to protect sensitive data.  

---
## 📊 MLflow Experiment Tracking
All training metrics (loss, accuracy, memory usage) are logged in **MLflow**.
```bash
mlflow ui
```
Open **`http://localhost:5000`** to visualize training performance.

---
## 🤝 Contributing
Want to improve this project? Feel free to **fork & submit a pull request**!  

---
## 📜 License
This project is licensed under the **MIT License**.

---
## 🔗 References
- [Flower Federated Learning](https://flower.ai/)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [PyTorch Pruning](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [Flower Custom Client](https://flower.ai/docs/framework/tutorial-series-customize-the-client-pytorch.html)
