# 📦 Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 🎨 Set plot style
sns.set(style="whitegrid")
plt.style.use("ggplot")

# 📂 Load your dataset
file_path = "your_file.csv"  # 🔁 Replace this with your actual file path
df = pd.read_csv(file_path)

# 📌 Basic info
print("🔷 Shape:", df.shape)
print("🔷 Columns:", df.columns.tolist())
print("🔷 Data Types:\n", df.dtypes)
print("🔷 Missing Values:\n", df.isnull().sum())
print("🔷 Duplicates:", df.duplicated().sum())
print("🔷 Summary Stats:\n", df.describe(include='all'))

# 📈 Mean, Median, Mode
numeric_cols = df.select_dtypes(include=[np.number])
mean_values = numeric_cols.mean()
median_values = numeric_cols.median()
mode_values = numeric_cols.mode().iloc[0]  # First mode row

print("\n📊 Mean:\n", mean_values)
print("\n📊 Median:\n", median_values)
print("\n📊 Mode:\n", mode_values)

# 🔥 Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# 📊 Histograms
for col in numeric_cols.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Histogram of {col}")
    plt.show()

# 📦 Boxplots
for col in numeric_cols.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# 🔤 Categorical Value Counts
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    print(f"\n🔠 Value Counts for '{col}':\n", df[col].value_counts())

# 🔁 Pairplot (if not too many columns)
if numeric_cols.shape[1] <= 6:
    sns.pairplot(df[numeric_cols.columns].dropna())
    plt.show()










import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load user dataset
file_path = input("Enter the path to your CSV file: ")
df = pd.read_csv(file_path)

print("\nColumns in your dataset:")
print(df.columns)

# Step 2: Let user choose X and Y columns
x_column = input("\nEnter the name of the column to use as independent variable (X): ")
y_column = input("Enter the name of the column to use as dependent variable (Y): ")

# Step 3: Prepare the data
X = df[[x_column]]  # Must be 2D
y = df[y_column]

# Step 4: Train the model
model = LinearRegression()
model.fit(X, y)

# Step 5: Display results
print("\nModel Coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[0]}")

# Step 6: Predict values
y_pred = model.predict(X)

# Step 7: Calculate R² and Mean Squared Error
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

print(f"\nR² (Coefficient of Determination): {r2}")
print(f"Mean Squared Error (MSE): {mse}")

# Step 8: Plot the data and regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.title('Linear Regression')
plt.legend()
plt.grid(True)
plt.show()





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Load dataset from user input
file_path = input("Enter path to your CSV file: ")
data = pd.read_csv(file_path)

# Show dataset preview
print("\nDataset Preview:")
print(data.head())

# Let user define the target column
target_column = input("\nEnter the target (label) column name: ")

# Split features and label
X = data.drop(columns=[target_column])
y = data[target_column]

# Encode categorical features if necessary
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred) * 100
print("\nAccuracy Score: {:.2f}%".format(accuracy))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot feature importances (if all features are numeric)
if X.select_dtypes(include=['number']).shape[1] == X.shape[1]:
    coef = model.coef_[0]
    features = X.columns
    plt.figure(figsize=(8, 5))
    plt.barh(features, coef)
    plt.xlabel('Coefficient Value')
    plt.title('Feature Importance (Logistic Regression)')
    plt.tight_layout()
    plt.show()
else:
    print("\nFeature importance plot skipped due to non-numeric features.")







import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
file_path = 'your_dataset.csv'  # 🔁 Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Step 2: Preprocess the data
# 🔁 Replace 'target' with the name of your label column
X = df.drop('target', axis=1)
y = df['target']

# Convert categorical columns to numeric if necessary
X = pd.get_dummies(X)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Step 7: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Step 8: Plot the Confusion Matrix as a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 9: Visualize the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=X.columns, class_names=[str(cls) for cls in model.classes_], filled=True)
plt.title("Decision Tree Visualization")
plt.show()





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, r2_score
import os

# Load your dataset
file_path = input("Enter the path to your CSV file: ")
if not os.path.exists(file_path):
    print("File not found. Please check the path.")
    exit()

df = pd.read_csv(file_path)
print("\nColumns in your dataset:\n", df.columns.tolist())

# Select target column
target_column = input("\nEnter the name of the target column: ")

# Feature and target split
X = df.drop(columns=[target_column])
y = df[target_column]

# Check if classification or regression
if y.dtypes == 'object' or len(y.unique()) <= 20:
    problem_type = 'classification'
else:
    problem_type = 'regression'

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate
if problem_type == 'classification':
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
else:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nR² Score:", r2_score(y_test, y_pred))

# Optional: Feature importance
import matplotlib.pyplot as plt
import seaborn as sns

feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()









import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset from user-given CSV file
file_path = input("Enter the path to your CSV file: ")
df = pd.read_csv(file_path)

print("\nDataset Head:")
print(df.head())

# Ask user which column is the target (label)
target_column = input("\nEnter the name of the target column: ")

# Prepare features (X) and target (y)
X = df.drop(columns=[target_column])
y = df[target_column]

# Handle categorical features (if any)
X = pd.get_dummies(X)

# Split data into training and testing sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))









import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset
file_path = input("Enter the path to your dataset (CSV file): ")
df = pd.read_csv(file_path)

# Step 2: Preview the dataset
print("\nColumns in dataset:")
print(df.columns)
features = input("Enter column names to use for clustering (comma separated): ").split(',')

# Step 3: Extract features for clustering
X = df[features].dropna()

# Step 4: Scale the data (optional but recommended)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Ask for number of clusters
k = int(input("Enter number of clusters (k): "))

# Step 6: Apply KMeans
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Step 7: Add cluster labels to original data
df['Cluster'] = labels
print("\nClustered Data Preview:")
print(df.head())

# Step 8: Visualize (only if 2D)
if len(features) == 2:
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
    plt.title("K-Means Clustering")
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.show()
else:
    print("Visualization works best with 2 features. You selected more than 2.")

# Optional: Save result
save = input("Do you want to save the clustered data to a new CSV file? (yes/no): ").lower()
if save == "yes":
    output_file = input("Enter output file name (with .csv): ")
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")










# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, silhouette_score

# Load dataset
df = pd.read_csv('data.csv')

print("🔍 First 5 rows of the dataset:")
print(df.head())

# ---- 1. EDA (Exploratory Data Analysis) ----
print("\n📊 Dataset Info:")
print(df.info())

print("\n📈 Summary Statistics:")
print(df.describe())

print("\n🧹 Checking for missing values:")
print(df.isnull().sum())

sns.pairplot(df.select_dtypes(include=np.number))
plt.show()

# ---- Select features and target ----
target_column = 'target'  # Change this to your actual target column name
X = df.drop(target_column, axis=1)
y = df[target_column]

# Split for supervised models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- 2. Linear Regression (if target is numeric) ----
if pd.api.types.is_numeric_dtype(y):
    print("\n📐 Linear Regression:")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("MSE:", mean_squared_error(y_test, y_pred))
else:
    print("\n📐 Linear Regression skipped: Target is not numeric.")

# ---- 3. Logistic Regression (for binary classification) ----
if y.nunique() == 2:
    print("\n⚖ Logistic Regression:")
    logr = LogisticRegression(max_iter=1000)
    logr.fit(X_train, y_train)
    y_pred = logr.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
else:
    print("\n⚖ Logistic Regression skipped: Target is not binary.")

# ---- 4. Decision Tree Classifier ----
print("\n🌲 Decision Tree Classifier:")
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---- 5. Random Forest Classifier ----
print("\n🌳 Random Forest Classifier:")
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---- 6. Naive Bayes ----
print("\n🧠 Naive Bayes Classifier:")
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---- 7. K-Means Clustering ----
print("\n📍 K-Means Clustering:")
k = 3  # You can change based on your dataset
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X)

df['Cluster'] = clusters
print("Silhouette Score:", silhouette_score(X, clusters))

# Visualize clusters (if 2D)
if X.shape[1] == 2:
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis')
    plt.title("K-Means Clustering")
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.show()


