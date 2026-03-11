# 1. Import the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 2. Load the Dataset
df = pd.read_csv('Sales_Data.csv')

# 3. Preprocessing (Data Transformation)
# Logistic Regression needs numbers, not text! 
# Let's map 'Retail' to 0 and 'Wholesale' to 1
df['CustomerType_Binary'] = df['CustomerType'].map({'Retail': 0, 'Wholesale': 1})

# Define our Features (X) and Target (y)
X = df[['Quantity', 'TotalPrice']]
y = df['CustomerType_Binary']

# 4. Split the Dataset (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build and Train the Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# 6. Make Predictions on the Test set
predictions = logistic_model.predict(X_test)

# 7. Evaluate the Model
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print("--- Logistic Regression Results ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)