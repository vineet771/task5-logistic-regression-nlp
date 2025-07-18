import pandas as pd
import re
import nltk
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from sklearn.exceptions import UndefinedMetricWarning

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Suppress UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Expanded Sample Data (10 samples)
data = {
    'Review Text': [
        "I love this product, it works great!",
        "Terrible service, very disappointed",
        "Great quality and fast shipping",
        "Worst experience ever",
        "Excellent item, very happy",
        "Do not recommend this at all",
        "Absolutely fantastic!",
        "Very poor build and noisy",
        "Highly recommended",
        "Not worth the price"
    ],
    'Sentiment': [
        'positive', 'negative', 'positive', 'negative', 'positive',
        'negative', 'positive', 'negative', 'positive', 'negative'
    ]
}

df = pd.DataFrame(data)

# Text Preprocessing Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation/special chars
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned'] = df['Review Text'].apply(clean_text)

# Encode Target
le = LabelEncoder()
y = le.fit_transform(df['Sentiment'])  # 0 = negative, 1 = positive
X = df['cleaned']

# Pipeline: TF-IDF + Logistic Regression
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# Train the Model
model.fit(X_train, y_train)

# Predict and Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))