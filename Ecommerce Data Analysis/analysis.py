import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv("sales_data (1).csv")

# Clean column names
df.columns = df.columns.str.strip()

# 🔍 DEBUG (IMPORTANT)
print("Columns:", df.columns.tolist())

# Handle missing values
df.dropna(inplace=True)

# Convert date
df['Date'] = pd.to_datetime(df['Date'])

# 🔥 AUTO-DETECT revenue column
possible_cols = ['Total', 'Revenue', 'Sales', 'Total_Sales']

revenue_col = None
for col in possible_cols:
    if col in df.columns:
        revenue_col = col
        break

if revenue_col is None:
    raise Exception("No revenue column found!")

print("Using column:", revenue_col)

# Calculations
total_revenue = df[revenue_col].sum()
best_product = df.groupby('Product')[revenue_col].sum().idxmax()
monthly_sales = df.groupby(df['Date'].dt.month)[revenue_col].sum()

# Create visuals folder
os.makedirs("visuals", exist_ok=True)

# Bar chart
df.groupby('Product')[revenue_col].sum().plot(kind='bar')
plt.title("Sales by Product")
plt.savefig("visuals/bar_chart.png")
plt.show()

# Line chart
monthly_sales.plot(kind='line', marker='o')
plt.title("Monthly Revenue Trend")
plt.savefig("visuals/line_chart.png")
plt.show()  