# 1. Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Load the dataset
df = pd.read_csv('Sales_Data.csv')

# 3. Encode the target variable: CustomerType (Retail=0, Wholesale=1)
df['CustomerType'] = df['CustomerType'].map({'Retail': 0, 'Wholesale': 1})

# 4. Feature engineering: add Year and Month from Date
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# 5. Define Features (X) and Target (y)
features = ['Quantity', 'UnitPrice', 'Discount', 'Year', 'Month']
categorical = ['Region', 'Product', 'StoreLocation', 'Salesperson', 'PaymentMethod']

# Encode categorical variables
for col in categorical:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    features.append(col)

X = df[features]
y = df['CustomerType']

# 6. Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 7. Split the data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Initialize and Train the Logistic Regression Model
logreg = LogisticRegression(max_iter=500)
logreg.fit(X_train, y_train)

# 9. Make Predictions on the test data
y_pred = logreg.predict(X_test)

# 10. Evaluate the Model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))