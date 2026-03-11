import pandas as pd
import matplotlib.pyplot as plt

# Load the coffee sales data
df = pd.read_csv('Coffee_sales.csv')

# --- 1. Visualization: Total Revenue by Coffee Type ---
# Group by coffee name and sum the money, then sort by revenue (descending)
coffee_revenue = df.groupby('coffee_name')['money'].sum().sort_values(ascending=False).reset_index()

plt.bar(coffee_revenue['coffee_name'], coffee_revenue['money'], color='skyblue')
plt.title('Total Revenue by Coffee Type')
plt.xlabel('Coffee Name')
plt.ylabel('Total Revenue ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('coffee_revenue.png')
plt.close()

# --- 2. Visualization: Total Revenue by Day of Week ---
# Group by Weekday and Weekdaysort to ensure bars are in chronological order
weekday_revenue = df.groupby(['Weekday', 'Weekdaysort'])['money'].sum().reset_index().sort_values('Weekdaysort')

plt.bar(weekday_revenue['Weekday'], weekday_revenue['money'], color='orange')
plt.title('Total Revenue by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Total Revenue ($)')
plt.tight_layout()
plt.savefig('weekday_revenue.png')
plt.close()

print("Visualizations saved as 'coffee_revenue.png' and 'weekday_revenue.png'.")