# ================================
# 📊 Sales Data Analysis Project
# ================================

import pandas as pd

# ================================
# Step 1: Load Dataset
# ================================
df = pd.read_csv("sales_data.csv")

# ================================
# Step 2: Explore Data
# ================================
print("🔍 First 5 rows:\n", df.head())
print("\n📐 Shape:", df.shape)
print("\n📄 Info:")
print(df.info())

# ================================
# Step 3: Data Cleaning
# ================================

# Clean column names
df.columns = df.columns.str.strip()

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values (better approach)
df["Total_Sales"].fillna(df["Total_Sales"].mean(), inplace=True)

# ================================
# Step 4: Data Analysis
# ================================

# Basic Metrics
total_revenue = df["Total_Sales"].sum()
average_sales = df["Total_Sales"].mean()
max_sales = df["Total_Sales"].max()
min_sales = df["Total_Sales"].min()

# Best Selling Product
best_product = df.groupby("Product")["Total_Sales"].sum().idxmax()

# ================================
# Step 5: Report (Console)
# ================================
print("\n" + "="*50)
print("📊 SALES ANALYSIS REPORT")
print("="*50)

print(f"💰 Total Revenue       : ₹{total_revenue:,.2f}")
print(f"📦 Best Product        : {best_product}")
print(f"📈 Average Sales       : ₹{average_sales:,.2f}")
print(f"🔝 Maximum Sales       : ₹{max_sales:,.2f}")
print(f"🔻 Minimum Sales       : ₹{min_sales:,.2f}")

print("="*50)

# ================================
# Step 6: Save Report to File
# ================================
with open("analysis_report.txt", "w") as f:
    f.write("SALES ANALYSIS REPORT\n")
    f.write("="*40 + "\n")
    f.write(f"Total Revenue: ₹{total_revenue:,.2f}\n")
    f.write(f"Best Product: {best_product}\n")
    f.write(f"Average Sales: ₹{average_sales:,.2f}\n")
    f.write(f"Max Sales: ₹{max_sales:,.2f}\n")
    f.write(f"Min Sales: ₹{min_sales:,.2f}\n")

print("\n✅ Report saved as 'analysis_report.txt'")