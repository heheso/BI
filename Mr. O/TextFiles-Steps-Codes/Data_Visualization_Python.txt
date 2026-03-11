import pandas as pd
import matplotlib.pyplot as plt

# Load the coffee sales data
df = pd.read_csv('Coffe_sales.csv')

# --- 1. Visualization: Total Revenue by Coffee Type ---
# Group by coffee name, sum money, and sort descending
coffee_revenue = df.groupby('coffee_name')['money'].sum().sort_values(ascending=False).reset_index()

plt.figure(figsize=(8,5))
plt.bar(coffee_revenue['coffee_name'], coffee_revenue['money'], color='skyblue')
plt.title('Total Revenue by Coffee Type')
plt.xlabel('Coffee Name')
plt.ylabel('Total Revenue ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('coffee_revenue.png')
plt.show()
plt.close()

# --- 2. Visualization: Total Revenue by Day of Week ---
# Group by Weekday and sort using Weekdaysort
weekday_revenue = df.groupby(['Weekday','Weekdaysort'])['money'].sum().reset_index()
weekday_revenue = weekday_revenue.sort_values('Weekdaysort')

plt.figure(figsize=(8,5))
plt.bar(weekday_revenue['Weekday'], weekday_revenue['money'], color='orange')
plt.title('Total Revenue by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Total Revenue ($)')
plt.tight_layout()
plt.savefig('weekday_revenue.png')
plt.show()
plt.close()

print("Visualizations saved as 'coffee_revenue.png' and 'weekday_revenue.png'.")