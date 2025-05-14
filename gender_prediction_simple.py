import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os

# Load the combined dataset
try:
    df = pd.read_csv(r'C:\Users\dhanu\Downloads\reduced nlp\combined_gender_text.csv', encoding='utf-8')
    print("Dataset head:")
    print(df.head())
    print("\nDataset shape:", df.shape)
except FileNotFoundError:
    print("File not found. Please make sure the file exists and the path is correct.")
    exit()  # Exit the script if the file is not found
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Preprocess the data (handle missing values, etc.)
df.dropna(subset=['text', 'gender'], inplace=True)  # Remove rows with missing text or gender
df = df[df['gender'].isin(['male', 'female'])]  # Keep only male/female

print("\nGender distribution after preprocessing:")
print(df['gender'].value_counts())

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['gender']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
print("\nTraining the model...")
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='female')  # Example: Positive label as female
recall = recall_score(y_test, y_pred, pos_label='female')
f1 = f1_score(y_test, y_pred, pos_label='female')

print("\nModel Performance Metrics:")
print("-" * 50)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print("\nDetailed Classification Report:")
print("-" * 50)
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer directly in the current directory
print("\nSaving model and vectorizer...")
try:
    # Save model and vectorizer in the current directory
    joblib.dump(model, 'gender_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    print("Model and vectorizer saved successfully in the current directory!")
except Exception as e:
    print(f"Error saving model: {e}")

def predict_gender(text):
    """Predicts the gender based on the input text."""
    text_vectorized = vectorizer.transform([text])
    predicted_gender = model.predict(text_vectorized)[0]
    probabilities = model.predict_proba(text_vectorized)[0]
    confidence = probabilities.max() * 100
    return predicted_gender, confidence

# Example predictions
print("\nExample Predictions:")
print("-" * 50)
example_texts = [
    "I love coding and playing video games.",
    "Shopping for new shoes and dresses today!",
    "Working on my car in the garage.",
    "Baking cookies for the family.",
    "Just finished a great workout at the gym."
]

for text in example_texts:
    predicted_gender, confidence = predict_gender(text)
    print(f"\nText: {text}")
    print(f"Predicted gender: {predicted_gender}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    pass 