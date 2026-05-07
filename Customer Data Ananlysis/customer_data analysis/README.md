# 📊 Week 5: Customer Sales Analysis
### Advanced Data Manipulation with Pandas

> **Course:** Python for Data Analysis | Week 5 Project  
> **Author:** [Your Name]  
> **Dataset Period:** Full Year 2023

---

## 📁 Project Structure

```
week5-customer-sales-analysis/
├── customer_analysis.ipynb   ← Main Jupyter notebook (all analysis)
├── sales_data.csv            ← 100 rows of order-level sales data
├── customer_data.csv         ← 500 customer records with demographics
├── customer_churn.csv        ← 500-row churn dataset
├── analysis_report.pdf       ← Professional PDF report with charts
├── requirements.txt          ← Python dependencies
└── README.md                 ← This file
```

---

## 🎯 Project Goals

1. Identify the most valuable customers by lifetime value
2. Analyse monthly and quarterly revenue trends
3. Understand regional and segment-level performance
4. Build a comprehensive data visualisation dashboard
5. Deliver actionable business recommendations

---

## ⚙️ Setup Instructions

### Step 1 – Clone the repository
```bash
git clone https://github.com/<your-username>/week5-customer-sales-analysis.git
cd week5-customer-sales-analysis
```

### Step 2 – Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows
```

### Step 3 – Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 – Launch the notebook
```bash
jupyter notebook customer_analysis.ipynb
```

---

## 📋 Analysis Walkthrough (Day-by-Day)

| Day | Focus | Notebook Section |
|-----|-------|-----------------|
| Day 1 | Data Loading & Exploration | Section 2 |
| Day 2 | Data Cleaning & Preparation | Section 3 |
| Day 3 | Customer Analysis (CLV, Segments) | Sections 4–5 |
| Day 4 | Sales Pattern Analysis | Sections 4, 6 |
| Day 5 | Advanced Analysis (Pivot, Churn) | Sections 6–7 |
| Day 6 | Dashboard Creation (6 Charts) | Section 8 |
| Day 7 | Report & Insights | Section 9 |

---

## 📊 Dataset Descriptions

### `sales_data.csv` — 100 rows × 11 columns
| Column | Type | Description |
|--------|------|-------------|
| order_id | string | Unique order identifier |
| customer_id | string | Foreign key to customer_data |
| product | string | Product name |
| category | string | Product category |
| quantity | int | Units ordered |
| unit_price | float | Price per unit ($) |
| discount | float | Discount rate (0–0.20) |
| revenue | float | Final revenue after discount |
| order_date | date | Order date (YYYY-MM-DD) |
| month | string | Month name |
| quarter | string | Fiscal quarter (Q1–Q4) |

### `customer_data.csv` — 500 rows × 8 columns
| Column | Type | Description |
|--------|------|-------------|
| customer_id | string | Unique customer ID |
| customer_name | string | Full name |
| email | string | Contact email |
| region | string | Sales region |
| segment | string | Customer tier (Premium/Regular/Basic) |
| age | int | Customer age |
| join_date | date | Date of first purchase |
| is_churned | int | 1 = churned, 0 = active |

### `customer_churn.csv` — 500 rows × 5 columns
| Column | Type | Description |
|--------|------|-------------|
| customer_id | string | Unique customer ID |
| name | string | Full name |
| region | string | Sales region |
| churned | int | 1 = churned, 0 = active |
| tenure_months | int | Months as a customer |

---

## 🧠 Key Pandas Concepts Demonstrated

- **Groupby & Aggregation** — `groupby().agg()` with multiple metrics
- **Multi-condition Filtering** — AND (`&`) and OR (`|`) boolean masks
- **String Operations** — `.str.strip()`, `.str.title()`, `.str.upper()`
- **Datetime Extraction** — `.dt.year`, `.dt.month`, `.dt.day_name()`
- **DataFrame Merging** — `pd.merge()` with `how='left'` and `how='inner'`
- **Pivot Tables** — `pd.pivot_table()` for cross-tabulation
- **Missing Values** — `fillna()` with median and string defaults

---

## 📈 Sample Output

```
=======================================================
     CUSTOMER SALES ANALYSIS REPORT – 2023
=======================================================
  Total Revenue      :      $95,477.98
  Total Orders       :             100
  Unique Customers   :              91
  Avg Order Value    :         $954.78
  Top Customer       : Nancy Joshi – $5,679.26
  Best Category      : Electronics
  Best Region        : East
=======================================================
```

---

## 💡 Business Recommendations

1. **Expand Electronics inventory** — accounts for 74%+ of total revenue
2. **Launch VIP loyalty programme** for top 10 customers to reduce churn risk
3. **Targeted regional campaigns** in underperforming regions (South, Central)
4. **Investigate Wearables Q4 gap** — zero revenue recorded in Q4
5. **Re-engagement campaign** for the 25% of customers showing churn risk

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Pandas | Data manipulation & analysis |
| NumPy | Numerical operations |
| Matplotlib | Base plotting |
| Seaborn | Statistical visualisations |
| Jupyter Notebook | Interactive development |
| ReportLab | PDF report generation |

---

## ✅ Submission Checklist

- [x] `customer_analysis.ipynb` — Complete notebook with all analysis
- [x] `sales_data.csv` — Sales dataset (100 rows)
- [x] `customer_data.csv` — Customer dataset (500 rows)
- [x] `customer_churn.csv` — Churn dataset (500 rows)
- [x] `analysis_report.pdf` — Professional PDF with 6 charts
- [x] `requirements.txt` — All dependencies listed
- [x] `README.md` — Full documentation

---

*Built with ❤️ using Python & Pandas*
