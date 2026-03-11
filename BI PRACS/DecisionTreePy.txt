# 1. Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 2. Load the dataset
df = pd.read_csv('Sales_Data.csv')

# 3. Prepare Features (X) and Target (y)
# We want to predict 'CustomerType' (Wholesale/Retail) 
# based on 'Quantity' and 'UnitPrice'
X = df[['Quantity', 'UnitPrice']]
y = df['CustomerType']

# 4. Split the data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize and Train the Decision Tree
# We set max_depth=3 so the tree doesn't get too messy to see
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

# 6. Visualize the Tree (Equivalent to R's plot() function)
plt.figure(figsize=(12, 8))
plot_tree(clf, 
          feature_names=['Quantity', 'UnitPrice'], 
          class_names=clf.classes_, 
          filled=True, 
          rounded=True)

plt.title("Decision Tree: Predicting Customer Type")
plt.show()