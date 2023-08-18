import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# Load the CSV file
input_file = "./0. combined/combined_data.csv"  # Replace 'your_input_file.csv' with the path to your CSV file
df = pd.read_csv(input_file)

# further processing the data, removing stopwords and tokenizing
stop_words = set(stopwords.words("english"))
df["processed_text"] = df["text"].apply(
    lambda x: " ".join([word for word in word_tokenize(x) if word not in stop_words])
)

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

# training model
model = SVC(
    C=10,
    class_weight=None,
    degree=2,
    gamma=0.1,
    kernel="rbf",
)
model.fit(X_train_tfidf, y_train)

# testing on validation set
y_pred = model.predict(X_val_tfidf)

# Calculate accuracy and other metrics
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_val, y_pred))
