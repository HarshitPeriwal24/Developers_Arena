# Executive Summary
## Business Analytics Capstone — June 2025

### Overview
This project delivers end-to-end data analysis across three critical business domains: Customer Churn (telecom), Residential Real Estate Pricing, and Sales Performance. Using Python, scikit-learn, and seaborn, we processed 900+ records, built predictive models, and translated statistical findings into 12 actionable business recommendations.

---

### Key Findings

**1. Customer Churn (10.6% base rate)**
- Month-to-month contract holders churn 3× more than annual subscribers
- New customers (tenure < 12 months) represent the highest risk cohort
- Random Forest model achieves ~90% accuracy in identifying at-risk customers
- **Projected savings:** Reducing churn to 8% retains ~13 additional customers per 500 monthly — significant at scale

**2. House Price Drivers**
- Area (sq ft) explains the largest share of price variance (R² ≈ 0.85 with Random Forest)
- Urban properties command a ~40% median premium over rural equivalents
- Villas outperform apartments of equivalent area by 15–20%
- Property age has a modest but statistically significant negative effect on price

**3. Sales Performance**
- Phones account for >40% of total revenue; protecting this category is critical
- South region outperforms all others; replicating its sales approach is the highest-ROI initiative
- ANOVA confirms regional revenue differences are statistically significant (p < 0.05)
- Q1 shows a consistent revenue dip — a targeted promotion window

---

### Top 3 Immediate Actions
1. **Launch contract upgrade campaign** with 10–15% discount for month-to-month customers (expected churn reduction: 2–3 pp within 90 days)
2. **Reallocate Q3 marketing budget**: +20% to South region replication, −10% from underperforming West
3. **Deploy ML churn scoring** to CRM — flag customers with predicted churn probability > 60% for proactive outreach

---

### Technical Stack
Python 3.11 · pandas · scikit-learn · seaborn · matplotlib · scipy · Jupyter

*Full analysis: `notebooks/capstone_analysis.ipynb` | Visualisations: `reports/fig1–fig6`*
