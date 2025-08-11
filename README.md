# 🫀 Heart Disease Prediction App

This is a **machine learning-powered Streamlit web application** that predicts whether a person has heart disease based on their medical data.  
The model takes input parameters like age, cholesterol, blood pressure, and more, and outputs a **prediction with probability**.

## 📌 Features
- Upload **patient CSV file** to get predictions.
- Shows whether heart disease is **detected or not**, along with the prediction probability.
- Works with **single patient files** or multiple patient datasets.
- Includes **sample Indian patient files** for testing.

## 📂 Project Structure
.
├── heart.csv                # Original dataset used for training  
├── model.pkl                # Saved ML model  
├── app.py                   # Streamlit application  
├── sample_patients/         # Example single-patient CSV files  
└── README.md                # Project documentation  

## ⚙️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
