import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df_sal = pd.read_csv('D:\Desktop\ML Assignment 2\Salary_Data.csv')

# Display the first five rows of the dataset
print("First five rows of the dataset:")
print(df_sal.head())

# Display a concise summary of the DataFrame
print("\nDataFrame Information:")
print(df_sal.info())

# Display descriptive statistics
print("\nDescriptive statistics:")
print(df_sal.describe())

# Check for missing values
print("\nChecking for missing values:")
print(df_sal.isnull().sum())

# Display the data types of each column
print("\nData types of each column:")
print(df_sal.dtypes)

# Display the number of unique values in each column
print("\nNumber of unique values in each column:")
print(df_sal.nunique())

# Display the correlation matrix
print("\nCorrelation matrix:")
print(df_sal.corr())

# Plot the distribution of 'Salary' using histplot
plt.title('Salary Distribution Plot')
sns.histplot(df_sal['Salary'], kde=True)
plt.show()

# Plot the distribution of 'YearsExperience' using histplot
plt.title('Years of Experience Distribution Plot')
sns.histplot(df_sal['YearsExperience'], kde=True)
plt.show()

# Scatter plot of YearsExperience vs. Salary
plt.scatter(df_sal['YearsExperience'], df_sal['Salary'], color='lightcoral')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.box(False)
plt.show()

# Define independent (X) and dependent (y) variables
X = df_sal[['YearsExperience']]  # independent variable
y = df_sal['Salary']             # dependent variable

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize and fit the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred_test = regressor.predict(X_test)     # predicted values of y_test
y_pred_train = regressor.predict(X_train)   # predicted values of y_train

# Plotting the Training set results
plt.scatter(X_train, y_train, color='lightcoral')
plt.plot(X_train, y_pred_train, color='firebrick')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['Regression Line', 'Actual Salary'], loc='best', facecolor='white')
plt.box(False)
plt.show()

# Plotting the Test set results
plt.scatter(X_test, y_test, color='lightcoral')
plt.plot(X_train, y_pred_train, color='firebrick')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['Regression Line', 'Actual Salary'], loc='best', facecolor='white')
plt.box(False)
plt.show()

# Regressor coefficients and intercept
print(f'Coefficient: {regressor.coef_[0]}')
print(f'Intercept: {regressor.intercept_}')

# Evaluating the model
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f'\nMean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
