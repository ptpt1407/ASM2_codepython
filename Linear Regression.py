import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'data_BSP.csv'
data = pd.read_csv(file_path)

# Drop rows with missing target values
data_cleaned = data.dropna(subset=['total_amount'])

# Select features and target
features = ['quantity', 'discount']
X = data_cleaned[features]
y = data_cleaned['total_amount']

# Handle missing values in features
X.fillna(X.mean(), inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the TotalAmount for the test set
y_pred = model.predict(X_test)

# Plotting the first 20 predicted values
plt.figure(figsize=(10, 6))
plt.plot(range(20), y_pred[:20], marker='o', label='Predicted total_amount')
plt.title('Predicted total_amount (First 20 values)')
plt.xlabel('Index')
plt.ylabel('total_amount')
plt.legend()
plt.grid(True)
plt.show()