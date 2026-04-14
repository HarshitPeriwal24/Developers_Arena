# Sales Data Analysis Project
import pandas as pd
df = pd.read_csv("sales_data.csv")

# Step 2: Explore Data
print("🔍 First 5 rows:")
print(df.head())

print("\n Dataset Info:")
print(df.info())

print("\n Shape of Data:", df.shape)

# Step 3: Data Cleaning
# Remove duplicate rows
df = df.drop_duplicates()

# Handle missing values
df = df.fillna(0)

# Step 4: Data Analysis
# Total Revenue
total_revenue = df["Total_Sales"].sum()

# Best Selling Product
best_product = df.groupby("Product")["Total_Sales"].sum().idxmax()

# Average Sales
average_sales = df["Total_Sales"].mean()

# Step 5: Report
print("\n" + "="*50)
print("📊 SALES ANALYSIS REPORT")
print("="*50)

print(f" Total Revenue       : ₹{total_revenue:,.2f}")
print(f" Best Selling Product: {best_product}")
print(f" Average Sales       : ₹{average_sales:,.2f}")

print("="*50)