
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


df = pd.read_csv(r'D:\Desktop\ML LAB\ML Assignment 3\framingham.csv')

print(df.head())
print(df.info())  # Check for missing values and datatypes
print(df.describe())  # Summary statistics
print(df.isnull().sum())  # Check for missing values

df.dropna(inplace=True)  # Remove rows with missing values
# OR
df.fillna(df.mean(), inplace=True)  # Fill missing values with the column mean

X = df.drop(columns=['TenYearCHD'])  # Features (all columns except the target)
y = df['TenYearCHD']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


model = LogisticRegression(C=0.1, max_iter=500, solver='liblinear')
model.fit(X_train, y_train)
