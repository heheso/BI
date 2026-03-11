# --- Simple Sales Data Analysis in Python ---

import pandas as pd

# Step 1: Load the CSV file
df = pd.read_csv('sales_data.csv')

# Step 2: Quick look at the data
print("First 5 rows of the dataset:")
print(df.head())

# Step 3: Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Step 4: Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Step 5: Basic statistics
print("\nBasic statistics:")
print(df[['Quantity', 'UnitPrice', 'TotalPrice']].describe())

# Step 6: Total sales and total quantity
total_sales = df['TotalPrice'].sum()
total_quantity = df['Quantity'].sum()
print(f"\nTotal Sales: ${total_sales:.2f}")
print(f"Total Quantity Sold: {total_quantity}")

# Step 7: Top 5 Products by Total Sales
top_products = df.groupby('Product')['TotalPrice'].sum().sort_values(ascending=False).head(5)
print("\nTop 5 Products by Sales:")
print(top_products)

# Step 8: Sales by Region
sales_region = df.groupby('Region')['TotalPrice'].sum().sort_values(ascending=False)
print("\nTotal Sales by Region:")
print(sales_region)

# Step 9: Sales by Customer Type
sales_customer = df.groupby('CustomerType')['TotalPrice'].sum()
print("\nTotal Sales by Customer Type:")
print(sales_customer)

# Step 10: Visualization (optional)
import matplotlib.pyplot as plt
import seaborn as sns

# Total Sales by Product
plt.figure(figsize=(8,5))
sns.barplot(x=top_products.index, y=top_products.values, palette='gray')
plt.title('Top 5 Products by Total Sales')
plt.xlabel('Product')
plt.ylabel('Total Sales ($)')
plt.tight_layout()
plt.show()

# Sales by Region
plt.figure(figsize=(8,5))
sns.barplot(x=sales_region.index, y=sales_region.values, palette='gray')
plt.title('Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales ($)')
plt.tight_layout()
plt.show()