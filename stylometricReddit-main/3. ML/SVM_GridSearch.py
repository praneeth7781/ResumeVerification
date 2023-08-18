import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file
input_file = "./0. combined/combined_data.csv"  # Replace 'your_input_file.csv' with the path to your CSV file
df = pd.read_csv(input_file)

# further processing the data, removing stopwords
stop_words = set(stopwords.words("english"))
df["processed_text"] = df["text"].apply(
    lambda x: " ".join([word for word in word_tokenize(x) if word not in stop_words])
)

# tokenizing
X = df["processed_text"]
y = df["author"]

# splitting into training and validation sets, 80% t, 20% v
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=1227
)

# vectorizing data using TFIDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

# Set up the parameter grid for hyperparameter tuning
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
    "degree": [2, 3, 4, 5],
    "class_weight": [None, "balanced", {0: 1, 1: 2}],
}

# Create the SVM model
svm_model = SVC()

# Initialize GridSearchCV with the parameter grid and the SVM model
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5)

# Fit the GridSearchCV to the data to find the best hyperparameters
grid_search.fit(X_train_tfidf, y_train)

# Access the best hyperparameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Use the best model to make predictions and evaluate its performance
y_pred = best_model.predict(X_val_tfidf)

# Calculate accuracy and other metrics
accuracy = accuracy_score(y_val, y_pred)
print(f"Best Hyperparameters: {best_params}")
print(f"Best Model Accuracy: {accuracy:.2f}")
print(classification_report(y_val, y_pred))
