# 🚀 Week 6 Project: Interactive Sales Dashboard

> **Data Visualization Mastery with Seaborn & Plotly**

---

## 📋 Project Overview

This project is a complete **Interactive Sales Dashboard** built as part of Week 6 of the Data Science curriculum. It analyzes **100 sales transactions** across **5 products** and **4 regions** over January–April 2024. The dashboard uses **Seaborn** for publication-quality statistical plots and **Plotly** for fully interactive, browser-based visualizations.

### 🎯 Goals
- Understand sales trends across time, product, and region
- Identify top-performing products and customers
- Visualize data distributions and correlations using statistical plots
- Build an interactive, shareable dashboard

---

## 🗂️ Project Structure

```
sales_dashboard/
├── dashboard.ipynb          # Jupyter notebook (step-by-step walkthrough)
├── dashboard.py             # Main Python script (run all charts at once)
├── sales_data.csv           # Dataset (100 rows × 7 columns)
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── visualizations/
    ├── day1_seaborn_basics.png        # Bar + Line charts
    ├── day2_statistical.png           # Box + Violin plots
    ├── day3_heatmaps.png              # Correlation heatmaps
    ├── day4_multiplot.png             # 2×2 subplot grid
    ├── day5_interactive_line.html     # Plotly interactive line chart
    ├── day5_interactive_sunburst.html # Plotly sunburst chart
    ├── day5_interactive_scatter.html  # Plotly animated scatter
    ├── day6_full_dashboard.html       # 🌟 Full interactive dashboard
    └── day7_final_dashboard.png       # Polished static dashboard
```

---

## ⚙️ Setup Instructions

### 1. Clone / Download the project
```bash
git clone <your-repo-url>
cd sales_dashboard
```

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the dashboard
```bash
# Option A: Run all charts at once
python3 dashboard.py

# Option B: Open the Jupyter notebook
jupyter notebook dashboard.ipynb
```

### 5. View interactive dashboard
Open `visualizations/day6_full_dashboard.html` in any browser — no server needed!

---

## 📊 Dataset Description

**File:** `sales_data.csv`  
**Rows:** 100 | **Columns:** 7

| Column | Type | Description |
|--------|------|-------------|
| `Date` | datetime | Transaction date (Jan 1 – Apr 9, 2024) |
| `Product` | string | Phone, Laptop, Tablet, Headphones, Monitor |
| `Quantity` | int | Units ordered (1–9) |
| `Price` | int | Unit price in ₹ |
| `Customer_ID` | string | Unique customer identifier |
| `Region` | string | East, West, North, South |
| `Total_Sales` | int | Quantity × Price |

---

## 📅 Day-by-Day Guide

### Day 1 — Seaborn Basics
**File:** `visualizations/day1_seaborn_basics.png`

- **Bar chart:** Total revenue by product category
- **Line chart:** Monthly revenue trend with fill area
- **Key insight:** Laptops generate the most revenue; March had peak sales

### Day 2 — Statistical Visualizations
**File:** `visualizations/day2_statistical.png`

- **Box plot:** Price spread and outliers for each product — shows median, IQR, and extremes
- **Violin plot:** Full sales distribution by region — reveals whether distributions are symmetric or skewed
- **Key insight:** Laptops have the widest price range; North region has highest median sales

### Day 3 — Heatmaps & Correlation
**File:** `visualizations/day3_heatmaps.png`

- **Correlation matrix:** Shows strong positive correlation between Price and Total_Sales (expected), and weak correlation with Quantity
- **Pivot heatmap:** Average sales per Region × Product combination — identifies best-performing combinations
- **Key insight:** North + Laptop is the highest-revenue combination

### Day 4 — Multi-Plot 2×2 Dashboard
**File:** `visualizations/day4_multiplot.png`

- **Stacked bar:** Month-by-month breakdown by product
- **Count bar:** Number of transactions per region
- **Scatter plot:** Price vs Total Sales coloured by product
- **Horizontal bar:** Average quantity ordered per product

### Day 5 — Interactive Plotly Visualizations
**Files:** `visualizations/day5_interactive_*.html`

| Chart | Description |
|-------|-------------|
| `day5_interactive_line.html` | Hover over any point to see exact figures; toggle products on/off |
| `day5_interactive_sunburst.html` | Drill down from Region → Product; click to zoom |
| `day5_interactive_scatter.html` | Animated scatter: press ▶ to watch sales evolve month by month |

### Day 6 — Full Dashboard Integration
**File:** `visualizations/day6_full_dashboard.html`

A 3×2 Plotly dashboard combining:
1. Monthly sales trend (all products)
2. Revenue share pie/donut
3. Box plots for price distribution
4. Region bar chart
5. Region × Product heatmap
6. Top 10 customers by spend

### Day 7 — Polish & Final Dashboard
**File:** `visualizations/day7_final_dashboard.png`

Same 6-chart layout rendered as a high-resolution PNG with:
- Dark professional theme
- Consistent color palette
- Title banner and subtitle branding
- Ready for PDF report / presentation slides

---

## 🎨 Design Decisions

### Color Palette
```python
PALETTE = {
    'Phone':      '#4361EE',   # Blue
    'Laptop':     '#3A0CA3',   # Deep purple
    'Tablet':     '#7209B7',   # Violet
    'Headphones': '#F72585',   # Hot pink
    'Monitor':    '#4CC9F0',   # Cyan
}
```
- Chosen for maximum contrast on dark background
- Each product has a dedicated color used consistently across all charts

### Theme
- **Background:** `#0F0F1A` (deep navy) — reduces eye strain, looks professional
- **Card background:** `#1A1A2E`
- All Seaborn charts use `plt.rcParams` to apply the theme globally
- Plotly charts use `template='plotly_dark'` with matching hex colors

---

## 🧪 Testing & Validation

### Data validation checks (run in notebook)
```python
# Check for missing values
df.isnull().sum()             # Expected: all zeros

# Verify Total_Sales = Quantity × Price
assert (df['Total_Sales'] == df['Quantity'] * df['Price']).all()

# Check date range
print(df['Date'].min(), df['Date'].max())

# Check unique products match palette
assert set(df['Product'].unique()) == set(PALETTE.keys())
```

### Output file validation
```bash
ls -lh visualizations/
# Should show 9 files (5 PNG + 4 HTML)
```

---

## 📈 Key Business Insights

| Metric | Value |
|--------|-------|
| Total Revenue | ₹12.37 Million |
| Total Transactions | 100 |
| Avg Order Value | ₹1,23,650 |
| Top Product | Laptop |
| Top Region | North |
| Best Month | March 2024 |
| Highest Single Order | ₹3,73,932 |

---

## 📚 Resources Used

- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Plotly Python Docs](https://plotly.com/python/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [Color Theory for Data Viz](https://colorbrewer2.org/)

---

## 🏆 Quality Checklist

- [x] 5+ chart types (bar, line, box, violin, heatmap, scatter, pie, sunburst)
- [x] Seaborn for all statistical plots
- [x] Interactive Plotly charts with hover, animation, dropdown
- [x] Cohesive dark color scheme across all charts
- [x] Professional dashboard layout (2×3 grid)
- [x] Correlation heatmap
- [x] Box + Violin plots with annotations
- [x] Animated scatter plot
- [x] Sunburst drill-down chart
- [x] All outputs saved to `visualizations/` folder
- [x] Requirements.txt included
- [x] Jupyter notebook with explanations
- [x] README with setup instructions

---

*Week 6 — Data Visualization Mastery | Seaborn + Plotly*
