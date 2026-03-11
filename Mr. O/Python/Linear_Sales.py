# 1. Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 2. Load the real dataset
df = pd.read_csv('Sales_Data.csv')

# 3. Define Features (X) and Target (y)
# We use double brackets for X because scikit-learn expects a 2D array for features
X = df[['Quantity']] 
y = df['TotalPrice']

# 4. Split the data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Initialize and Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Make Predictions on the unseen test data
y_pred = model.predict(X_test)

# 7. Evaluate the Model's performance
r2 = r2_score(y_test, y_pred)
print(f"R-squared Score: {r2:.2f}")

# 8. Visualize the result (Crucial for BI presentations!)
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Predicting Total Price based on Quantity')
plt.xlabel('Quantity')
plt.ylabel('Total Price')
plt.legend()
plt.show()