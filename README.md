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

## 📂 Project Structure

```plaintext
gender-prediction-nlp/
├── model/
│   ├── best_model.pkl          # Trained model (Linear SVM)
│   └── vectorizer.pkl          # Saved TF-IDF vectorizer
├── train_model.py              # Model training and evaluation script
├── gui_app.py                  # GUI script for prediction
├── utils.py                    # Helper functions (optional)
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation
Linear SVM (Best)	0.7229	0.7246	0.7107	0.7176

✅ Best Model: Linear SVM based on F1-score

