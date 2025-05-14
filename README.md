# 🧠 Gender Prediction Using Text (NLP Project)

## 📌 Overview

This project focuses on **predicting gender (Male/Female)** based solely on text data using **Natural Language Processing (NLP)** and **Machine Learning** techniques. It employs traditional vectorization (TF-IDF) and machine learning classifiers like **Logistic Regression**, **Naive Bayes**, and **Support Vector Machine (SVM)** to achieve accurate classification results.

Additionally, a **Graphical User Interface (GUI)** using `Tkinter` allows users to input text and view gender predictions and sentiment analysis in real-time.

---

## 🎯 Project Goals

- Extract meaningful features from text using **TF-IDF**
- Train and compare multiple classifiers
- Identify the best-performing model
- Build a GUI-based application for real-time predictions
- Include sentiment analysis for additional context

---

## 🛠 Technologies Used

| Category         | Tools/Frameworks                  |
|------------------|-----------------------------------|
| Language         | Python 3                          |
| ML Libraries     | scikit-learn, pandas, joblib      |
| NLP              | TF-IDF, TextBlob, regex           |
| GUI              | Tkinter                           |
| File Management  | OS, datetime                      |

---

## 📂 Project Structure

gender-prediction-nlp/
├── model/
│ ├── best_model.pkl # Trained model (Linear SVM)
│ └── vectorizer.pkl # Saved TF-IDF vectorizer
├── train_model.py # Model training and evaluation script
├── gui_app.py # GUI script for prediction
├── utils.py # Helper functions (optional modularization)
├── README.md # Project documentation
├── requirements.txt # Python package dependencies

Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Train the Model

bash
Copy
Edit
python train_model.py
Run the GUI App

bash
Copy
Edit
python gui_app.py
🧪 Model Training & Evaluation
Three machine learning models were trained and evaluated using metrics like Accuracy, Precision, Recall, and F1-score. Based on the results, Linear SVM performed the best.

Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	0.7165	0.7210	0.6978	0.7092
Naive Bayes	0.6882	0.6619	0.7577	0.7065
Linear SVM (Best)	0.7229	0.7246	0.7107	0.7176

✅ Best Model: Linear SVM based on F1-score

