# 🧠 Gender Prediction Using Text (NLP Project)

## 📌 Overview

This project focuses on **predicting gender (Male/Female)** based on text data using **Natural Language Processing (NLP)** and **Machine Learning** techniques. It uses traditional vectorization (TF-IDF) and classifiers like **Logistic Regression**, **Naive Bayes**, and **Support Vector Machine (SVM)** to achieve accurate classification.

A **Graphical User Interface (GUI)** using `Tkinter` allows users to input text and view gender predictions and sentiment analysis in real-time.

---

## 🎯 Project Goals

* Extract features using **TF-IDF**
* Train and evaluate multiple classifiers
* Identify the best-performing model
* Build a GUI for real-time predictions
* Include sentiment analysis for context

---

## 🛠 Technologies Used

| Category          | Tools/Frameworks                   |
| ----------------- | ---------------------------------- |
| Language          | Python 3                           |
| ML Libraries      | scikit-learn, pandas, joblib       |
| NLP               | TF-IDF, TextBlob, regex            |
| GUI               | Tkinter                            |
| File Management   | os, datetime                       |

---

## 📂 Project Structure

```
gender-prediction-nlp/
├── model/
│   ├── best_model.pkl          # Trained model (Linear SVM)
│   └── vectorizer.pkl          # Saved TF-IDF vectorizer
├── train_model.py              # Model training and evaluation script
├── gui_app.py                  # GUI script for predictions
├── utils.py                    # Helper functions (optional)
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation
```

---

## ⚙️ Installation Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/dhanushreddy19/Text-Based-Gender-Prediction-Using-NLP.git
cd Text-Based-Gender-Prediction-Using-NLP
```

### 2️⃣ Create a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate        # On Linux/Mac
venv\Scripts\activate          # On Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model

```bash
python train_model.py
```

### 5️⃣ Run the GUI Application

```bash
python gui_app.py
```
