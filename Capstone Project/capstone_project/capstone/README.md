# 📊 Week 8 Capstone: Real-World Business Analysis

> End-to-end data analysis project covering Customer Churn, House Price Prediction, and Sales Performance — with ML models, professional visualisations, and actionable business recommendations.

---

## 🗂 Project Structure

```
capstone/
├── data/
│   ├── customer_churn_raw.csv        # 500 rows, 9 cols
│   ├── house_prices_raw.csv          # 300 rows, 8 cols
│   ├── sales_data_raw.csv            # 100 rows, 7 cols
│   ├── customer_churn_clean.csv      # Cleaned + engineered
│   ├── house_prices_clean.csv        # Cleaned + price features
│   └── sales_data_clean.csv          # Cleaned + seasonality
├── notebooks/
│   └── capstone_analysis.ipynb       # Main analysis notebook
├── reports/
│   ├── executive_summary.md          # 1-page business summary
│   ├── fig1_churn_eda.png
│   ├── fig2_house_eda.png
│   ├── fig3_sales_eda.png
│   ├── fig4_churn_model.png
│   ├── fig5_house_model.png
│   └── fig6_sales_advanced.png
├── presentation/
│   └── Business_Analytics_Capstone.pptx   # 11-slide deck
├── src/
│   └── data_cleaning.py              # Reusable cleaning module
├── requirements.txt
└── README.md
```

---

## 🎯 Business Questions Answered

| Domain | Question | Method |
|--------|----------|--------|
| Customer Churn | Which customers are most likely to leave? | Logistic Regression + Random Forest |
| House Prices | What factors drive property values? | Multiple Linear Regression + RF |
| Sales | Which products/regions generate the most revenue? | EDA + ANOVA + Trend Analysis |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/business-analytics-capstone.git
cd business-analytics-capstone
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run data cleaning
```bash
python src/data_cleaning.py
```

### 5. Open the main notebook
```bash
jupyter notebook notebooks/capstone_analysis.ipynb
```

---

## 📊 Analysis Techniques Used

1. **Descriptive Statistics** — Mean, median, distribution analysis across all datasets
2. **Exploratory Data Analysis** — 6 multi-panel figures with seaborn + matplotlib
3. **Logistic Regression** — Binary churn classification with StandardScaler
4. **Random Forest Classifier** — Feature importance + confusion matrix for churn
5. **Multiple Linear Regression** — House price baseline model
6. **Random Forest Regressor** — R² ≈ 0.85 for house price prediction
7. **One-Way ANOVA** — Tests whether regional sales differences are statistically significant
8. **Pearson Correlation** — Quantity vs Revenue relationship (r ≈ 0.80)
9. **Time-Series Analysis** — Monthly revenue trend + 3-month moving average
10. **Feature Engineering** — Tenure bins, price per sq ft, seasonality flags

---

## 💡 Key Insights

### Customer Churn
- **10.6%** overall churn rate; month-to-month customers churn **3×** more
- New customers (< 12 months) are the highest-risk cohort
- Random Forest achieves **~90% accuracy** on the test set
- **Recommendation:** Offer 10–15% discount for annual plan upgrades

### House Prices
- **Area (sq ft)** is the strongest predictor of price
- Urban properties command a **~40% premium** over rural
- Model R² ≈ **0.85** — explains 85% of price variance
- **Recommendation:** Segment listings by area band; prioritise urban pipeline

### Sales Performance
- **Phones** drive >40% of total revenue
- **South region** significantly outperforms others (ANOVA p < 0.05)
- Q1 shows a consistent revenue dip — ideal for promotional campaigns
- **Recommendation:** Replicate South region playbook; run Q1 bundle offers

---

## 📁 Visualisations

| Figure | Description |
|--------|-------------|
| `fig1_churn_eda.png` | Churn rate by contract, tenure, charges, senior status |
| `fig2_house_eda.png` | Price distributions, location comparison, correlation heatmap |
| `fig3_sales_eda.png` | Revenue by product/region, monthly trend, quarterly breakdown |
| `fig4_churn_model.png` | Random Forest confusion matrix + feature importances |
| `fig5_house_model.png` | Actual vs predicted prices + feature importances |
| `fig6_sales_advanced.png` | Revenue trend with moving average + product×region heatmap |

---

## 🗓 Implementation Roadmap

| Timeline | Action | KPI |
|----------|--------|-----|
| 30 days | Deploy churn scoring to CRM | Flag customers > 60% risk |
| 30 days | Contract upgrade campaign | +5 pp annual contract rate |
| 60 days | Onboarding email A/B test | -2 pp 90-day churn |
| 60 days | South-region playbook rollout | +15% West/North revenue |
| 90 days | Real-time churn dashboard | Weekly reporting cadence |
| 90 days | Automated sales variance report | Monthly stakeholder email |

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| pandas / numpy | Data manipulation |
| scikit-learn | ML models |
| seaborn / matplotlib | Visualisation |
| scipy | Statistical tests |
| Jupyter | Interactive notebooks |

---

## 📋 Submission Checklist

- [x] Minimum 500+ rows (customer_churn: 500 rows)
- [x] At least 3 analysis techniques (10 techniques used)
- [x] 5+ professional visualisations (6 multi-panel figures = 36 sub-plots)
- [x] Complete documentation (README + executive summary)
- [x] Business implementation plan (30/60/90 day roadmap)
- [x] Jupyter notebook with all code
- [x] PowerPoint presentation (11 slides)
- [x] requirements.txt for reproducibility

---

*Prepared by: Data Analytics Student · June 2025 · Week 8 Capstone Project*
