#============================
# Classification using Python
#============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================
# 1. Create synthetic dataset 
# ============================

data = {
    'Quantity': [14,1,14,18,18,2,7,3,3,5,15,19]*5,  
    'UnitPrice': [160,540,350,380,240,385,18,330,430,320,525,210]*5,
    'Discount': [0,0,0.1,0.15,0,0.15,0.1,0.15,0.05,0.1,0.1,0]*5,
    'Product': ['Laptop','Phone','Desk','Chair','Desk','Monitor','Chair','Chair','Tablet','Desk','Phone','Laptop']*5,
    'CustomerType': ['Wholesale','Retail','Wholesale','Wholesale','Retail','Wholesale','Retail','Wholesale','Wholesale','Retail','Retail','Wholesale']*5,
    'Region': ['East','South','North','Central','East','Central','South','Central','South','West','North','East']*5
}

df = pd.DataFrame(data)
print("Sample Dataset:\n", df.head(12))  

# ============================
# 2. Separate Features & Target
# ============================

numeric_features = ['Quantity','UnitPrice','Discount']
categorical_features = ['Product','CustomerType']

# One-hot encode categorical features
X_cat = pd.get_dummies(df[categorical_features])

# Combine numeric and categorical features
X = pd.concat([df[numeric_features], X_cat], axis=1)

# Target variable
y = df['Region']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ============================
# 3. Split Dataset
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=0, stratify=y_encoded
)

# ============================
# 4. Train Decision Tree Classifier
# ============================

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

# ============================
# 5. Make Predictions
# ============================

y_pred = model.predict(X_test)

# ============================
# 6. Evaluate Model
# ============================

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ============================
# Final Answer
# ============================
# Predicted values will be displayed.
# Accuracy of the Decision Tree classification model will also be shown.