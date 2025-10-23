# BrainSight-AI-Next-Gen-MRI-Based-Tumor-Detection-System
## ✨ Overview

This project applies **Deep Learning** to detect and classify brain tumors from MRI images.  
It combines **Convolutional Neural Networks (CNNs)** and **image preprocessing** techniques to achieve high accuracy.  
Includes a **Jupyter Notebook**, a **Python script** (`main.py`), and a **simple web interface** (`index.html`) for predictions.

<details>
<summary>🎯 <b>Key Objectives</b> (click to expand)</summary>

- Automate brain tumor detection using MRI scans  
- Build and train a CNN for image classification  
- Evaluate performance with accuracy, precision, recall, and F1-score  
- Provide a simple web-based interface for testing  
</details>

---

## 📂 Project Structure

```bash
📁 Brain-Tumor-Detection-Using-Deep-Learning
├── brain_tumour_detection_using_deep_learning.ipynb   # Jupyter notebook (training + analysis)
├── main.py                                            # Python script for training/inference
├── index.html                                         # Web demo interface
├── requirements.txt                                   # Dependencies
├── sample_images/                                     # Example MRI images
└── README.md                                         

Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate       # (Windows)
source venv/bin/activate    # (Mac/Linux)
Install dependencies
pip install -r requirements.txt

▶️ Usage
🧑‍💻 Run the Notebook

Launch Jupyter and open the notebook:

jupyter notebook brain_tumour_detection_using_deep_learning.ipynb

🧠 Run the Python Script

Train or predict directly:

python main.py --mode train --data_dir path/to/dataset
python main.py --mode predict --model path/to/model.h5 --image path/to/image.jpg

🌐 Run Web Demo

Open index.html in your browser to test the model with sample MRI images.

🧩 Dataset

You can use publicly available datasets such as:

🧠 Brain MRI Images for Brain Tumor Detection (Kaggle)

Example Folder Structure:
dataset/
 ├── training/
 │    ├── yes/
 │    └── no/
 └── testing/
      ├── yes/
      └── no/


Each folder contains MRI images labeled as Tumor or No Tumor.

🧠 Model Architecture

This project uses a Convolutional Neural Network (CNN) to extract spatial features from MRI scans.
Optionally, transfer learning (VGG16, ResNet50) can be applied for better performance.

📌 Workflow

Image loading and resizing

Data normalization and augmentation

CNN model training

Evaluation using validation and test data

Visualization of metrics and predictions

Loss Function: Binary Cross-Entropy
Optimizer: Adam
Evaluation Metrics: Accuracy, Precision, Recall, F1-score

📊 Results
Metric	Score
Accuracy	97%
Precision	96%
Recall	95%
F1-Score	95%

Sample Predictions:

MRI Input	Model Prediction

	🧠 Tumor

	✅ No Tumor
