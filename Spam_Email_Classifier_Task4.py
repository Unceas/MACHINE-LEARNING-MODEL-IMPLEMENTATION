# Spam Email Classifier - CodTech Internship Task 4

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: Load dataset
df = pd.read_csv(r"C:\Users\kayus\OneDrive\Desktop\Task4\spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Clean missing rows
df.dropna(subset=['label', 'message'], inplace=True)

# Show size before mapping
print("Before label mapping:", df.shape)

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Drop rows where mapping failed
df.dropna(subset=['label'], inplace=True)

# Convert to integer type
df['label'] = df['label'].astype(int)

# Show size after mapping
print("After label mapping:", df.shape)

# Debug output
print(df['label'].value_counts())


# STEP 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# STEP 6: Text vectorization
vectorizer = CountVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# STEP 7: Train the model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# STEP 8: Make predictions
y_pred = model.predict(X_test_vect)

# STEP 9: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# STEP 10: Sample prediction
sample = ["Congratulations! You have won a free ticket. Call now!"]
sample_vect = vectorizer.transform(sample)
print("\nSample Prediction:", "Spam" if model.predict(sample_vect)[0] else "Not Spam")
