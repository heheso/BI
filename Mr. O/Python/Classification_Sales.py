# 1. Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# 2. Load the dataset
df = pd.read_csv('Sales_Data.csv')

# 3. Encode the target variable: PaymentMethod (multi-class)
le_target = LabelEncoder()
df['PaymentMethod'] = le_target.fit_transform(df['PaymentMethod'])

# 4. Feature engineering: add Year and Month from Date
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# 5. Define Features (X) and Target (y)
features = ['Quantity', 'UnitPrice', 'Discount', 'Year', 'Month']
categorical = ['Region', 'Product', 'StoreLocation', 'CustomerType', 'Salesperson']

# Encode categorical variables
for col in categorical:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    features.append(col)

X = df[features]
y = df['PaymentMethod']

# 6. Split the data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Initialize and Train the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# 8. Make Predictions on the test data
y_pred = rf.predict(X_test)

# 9. Evaluate the Model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 10. Optional: Plot Feature Importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()