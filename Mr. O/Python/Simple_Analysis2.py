import pandas as pd

# --- Step 1: Read the data ---
df = pd.read_csv('Sales_Data.csv')

# --- Step 2: Check for missing values ---
if df.isnull().sum().sum() > 0:
    print("Warning: There are missing values in the dataset.")
else:
    print("No missing values detected.")

# --- Step 3: Calculate Total Sales and Profit ---
total_sales = df['TotalPrice'].sum()

# Estimate Profit as 20% of TotalPrice
df['Profit'] = df['TotalPrice'] * 0.20
total_profit = df['Profit'].sum()

# --- Step 4: Analysis by Product (as Category) ---
category_analysis = df.groupby('Product').agg({
    'TotalPrice': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
}).sort_values(by='TotalPrice', ascending=False)

# --- Step 5: Identify the Best Performing Product ---
best_product = category_analysis['TotalPrice'].idxmax()

# --- Step 6: Print Analysis Report ---
print("\n--- Sales and Profit Analysis Report ---")
print(f"Total Revenue Generated: ${total_sales:,.2f}")
print(f"Total Profit (Estimated 20%): ${total_profit:,.2f}")
print(f"Top-Selling Product: {best_product}\n")
print("Detailed Analysis per Product Category:")
print(category_analysis.round(2))