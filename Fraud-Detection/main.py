import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from collections import Counter

# Step 1: Load and Explore the Dataset
df = pd.read_csv('Fraud-Detection/creditcard.csv')

# Exploring the dataset
print(df.head())
print(df.info())

# Step 2: Data Preprocessing
# Check for missing values
print("Missing values count:", df.isnull().sum().max())

# Scaling amount column
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

# Step 3: Feature Selection/Engineering (if needed)
# No feature engineering for now, just using existing features

# Step 4: Model Selection and Training
# Splitting the dataset into train and test sets
X = df.drop(['Time', 'Class'], axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling class imbalance using SMOTE
print("Before SMOTE:", Counter(y_train))
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("After SMOTE:", Counter(y_train_res))

# Training Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_res, y_train_res)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Step 6: Visualization (ROC Curve)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
