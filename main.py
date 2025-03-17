import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression


# import our dataset
df = pd.read_csv("/Users/kvbiila/Downloads/Housing.csv")

# View the actual data in the set
print(df.head())

# view our different types of data in each column
print(df.info())

# convert our categorical values to numerical for easier exploration
df['mainroad'] = df['mainroad'].map({'yes': 1, 'no': 0})
df['guestroom'] = df['guestroom'].map({'yes': 1, 'no': 0})
df['basement'] = df['basement'].map({'yes': 1, 'no': 0})
df['hotwaterheating'] = df['hotwaterheating'].map({'yes': 1, 'no': 0})
df['airconditioning'] = df['airconditioning'].map({'yes': 1, 'no': 0})
df['prefarea'] = df['prefarea'].map({'yes': 1, 'no': 0})
df['furnishingstatus'] = df['furnishingstatus'].map({'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2})

print(df.head())
# Check missing data
missing = df.isnull().sum()
print(missing)

# handling of missing data if there is
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df_clean = df.dropna()

obj = pd.DataFrame({'col1': [1, None, 3, None, 5]})
obj.ffill(axis=None, inplace=True, limit=None)
obj.bfill(axis=None, inplace=True, limit=None)

# to visualise our housing dataset in a heatmap
df_heatmap = df.pivot_table(values='price', columns='bedrooms', aggfunc='mean')
# Plot the heatmap
plt.figure(figsize=(8, 6))  # Optional: Adjust the size of the heatmap
sns.heatmap(df_heatmap, annot=True, cmap="YlGnBu", fmt=".0f", linewidths=0.5, cbar=True)
plt.title('Heatmap of Average Price by Bedrooms')
plt.show()

# our histograms
plt.figure(figsize=(15, 10))  # Set the overall figure size

numeric_columns = df.select_dtypes(include=['number']).columns

n_cols = 3
n_rows = (len(numeric_columns) // n_cols) + int(len(numeric_columns) % n_cols > 0)  # Calculate rows required

# Create subplots with dynamic grid size
plt.figure(figsize=(15, n_rows * 5))  # Adjust the figure size based on the number of rows

# Loop through each numeric column to create histograms
for i, column in enumerate(numeric_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    plt.hist(df[column], bins=20, color='blue', edgecolor='black')
    plt.title(f'Histogram of {column.capitalize()}')
    plt.xlabel(column.capitalize())
    plt.ylabel('Frequency')
    plt.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

# statistical analysis
print(df.describe())

# preparing our data for machine learning model
# price is our target variable (y), and the other columns are features (X)
X = df_imputed.drop('price', axis=1)  # Features: all columns except 'price'
y = df_imputed['price']  # Target: 'price'

# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# predict and evaluate the model
y_predict = model.predict(X_test)

# Step 11: Display Results (optional)
print("\nPredictions on Test Set:\n", y_predict)
print("\nModel Score (R^2 on Test Set):", model.score(X_test, y_test))
