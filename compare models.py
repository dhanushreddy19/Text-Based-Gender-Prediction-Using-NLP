import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
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
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

# Preprocessing
df.dropna(subset=['text', 'gender'], inplace=True)
df = df[df['gender'].isin(['male', 'female'])]
print("\nGender distribution after preprocessing:")
print(df['gender'].value_counts())

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['gender']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='female')
    recall = recall_score(y_test, y_pred, pos_label='female')
    f1 = f1_score(y_test, y_pred, pos_label='female')

    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    print(f"{name} Performance:")
    print("-" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

# Select the best model based on F1-score
best_model_name = max(results, key=lambda name: results[name]['f1'])
best_model = models[best_model_name]
print(f"\nBest model selected: **{best_model_name}** based on F1-score.")

# Save best model and vectorizer
print("\nSaving best model and vectorizer...")
try:
    joblib.dump(best_model, 'gender_model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
    print("Model and vectorizer saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

# Gender prediction function
def predict_gender(text):
    text_vectorized = vectorizer.transform([text])
    predicted_gender = best_model.predict(text_vectorized)[0]
    try:
        probabilities = best_model.predict_proba(text_vectorized)[0]
        confidence = probabilities.max() * 100
    except AttributeError:
        confidence = 100  # SVMs don't return probabilities by default
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
