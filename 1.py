import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Sample Data
data = {
    'Study Hours': [5, 15, 9, 12, 2, 8, 6, 10, 7, 4],
    'Attendance': [70, 90, 80, 85, 40, 75, 60, 88, 65, 50],
    'Pass': [0, 1, 1, 1, 0, 1, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

# Data Exploration
print(df.describe())
print("Missing values:\n", df.isnull().sum())

# Plotting
sns.pairplot(df, hue='Pass')
plt.show()

# Train/Test Split
X = df[['Study Hours', 'Attendance']]
y = df['Pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
ConfusionMatrixDisplay(cm).plot()
plt.show()
