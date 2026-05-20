"""
=============================================================
  Week 6 Project: Interactive Sales Dashboard
  Author: Data Science Student
  Description: Multi-chart sales analysis dashboard using
               Seaborn (statistical plots) + Plotly (interactive)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings('ignore')

# ── Output folder ──────────────────────────────────────────
os.makedirs("visualizations", exist_ok=True)

# ══════════════════════════════════════════════════════════
# 1. LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════════

df = pd.read_csv("sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M').astype(str)
df['Month_Num'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
df['Day_Name'] = df['Date'].dt.day_name()

print("✅ Data loaded:", df.shape)
print(df.head(3))

# ══════════════════════════════════════════════════════════
# GLOBAL THEME
# ══════════════════════════════════════════════════════════

PALETTE = {
    'Phone':      '#4361EE',
    'Laptop':     '#3A0CA3',
    'Tablet':     '#7209B7',
    'Headphones': '#F72585',
    'Monitor':    '#4CC9F0',
}
REGION_PALETTE = ['#4361EE', '#F72585', '#7209B7', '#4CC9F0']
BG = '#0F0F1A'
CARD = '#1A1A2E'
TEXT = '#E8E8F0'
ACCENT = '#4361EE'

sns.set_theme(style='darkgrid', palette=list(PALETTE.values()))
plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   CARD,
    'axes.labelcolor':  TEXT,
    'xtick.color':      TEXT,
    'ytick.color':      TEXT,
    'text.color':       TEXT,
    'axes.titlecolor':  TEXT,
    'grid.color':       '#2A2A3E',
    'axes.edgecolor':   '#2A2A3E',
})

# ══════════════════════════════════════════════════════════
# DAY 1 — SEABORN BASICS: Bar + Line plots
# ══════════════════════════════════════════════════════════

print("\n📊 Day 1: Seaborn Basics...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(BG)
fig.suptitle("Sales Overview — Seaborn Basics", fontsize=18, fontweight='bold', color=TEXT, y=1.01)

# Bar chart: Total Sales by Product
product_sales = df.groupby('Product')['Total_Sales'].sum().sort_values(ascending=False)
colors = [PALETTE[p] for p in product_sales.index]
bars = axes[0].bar(product_sales.index, product_sales.values / 1e6, color=colors, edgecolor='white', linewidth=0.4)
axes[0].set_title("Total Sales by Product (₹ Millions)", fontsize=13, fontweight='bold', pad=12)
axes[0].set_xlabel("Product")
axes[0].set_ylabel("Total Sales (₹ Millions)")
for bar, val in zip(bars, product_sales.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'₹{val/1e6:.1f}M', ha='center', va='bottom', fontsize=9, color=TEXT)

# Line chart: Sales trend over time
monthly = df.groupby('Month')['Total_Sales'].sum().reset_index()
axes[1].plot(monthly['Month'], monthly['Total_Sales']/1e6, marker='o',
             color=ACCENT, linewidth=2.5, markersize=7, markerfacecolor='#F72585')
axes[1].fill_between(range(len(monthly)), monthly['Total_Sales']/1e6, alpha=0.15, color=ACCENT)
axes[1].set_xticks(range(len(monthly)))
axes[1].set_xticklabels(monthly['Month'], rotation=30, ha='right', fontsize=9)
axes[1].set_title("Monthly Sales Trend (₹ Millions)", fontsize=13, fontweight='bold', pad=12)
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Total Sales (₹ Millions)")

plt.tight_layout()
plt.savefig("visualizations/day1_seaborn_basics.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("   Saved → visualizations/day1_seaborn_basics.png")

# ══════════════════════════════════════════════════════════
# DAY 2 — STATISTICAL: Box + Violin Plots
# ══════════════════════════════════════════════════════════

print("📊 Day 2: Statistical Visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(BG)
fig.suptitle("Price & Sales Distribution — Statistical Plots", fontsize=18, fontweight='bold', color=TEXT)

# Box plot: Price distribution by Product
sns.boxplot(
    data=df, x='Product', y='Price',
    palette=PALETTE, width=0.5, linewidth=1.2,
    flierprops=dict(marker='o', markerfacecolor='#F72585', markersize=5, alpha=0.6),
    ax=axes[0]
)
axes[0].set_title("Price Distribution by Product", fontsize=13, fontweight='bold', pad=10)
axes[0].set_xlabel("Product")
axes[0].set_ylabel("Price (₹)")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x:,.0f}'))

# Violin plot: Total Sales by Region
sns.violinplot(
    data=df, x='Region', y='Total_Sales',
    palette=REGION_PALETTE, inner='quartile', linewidth=1.2,
    ax=axes[1]
)
axes[1].set_title("Total Sales Distribution by Region", fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel("Region")
axes[1].set_ylabel("Total Sales (₹)")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x/1e3:.0f}K'))

# Add median annotations
for i, region in enumerate(df['Region'].unique()):
    median = df[df['Region'] == region]['Total_Sales'].median()
    axes[1].text(i, median, f' Md:{median/1e3:.0f}K', va='center', fontsize=8, color=TEXT)

plt.tight_layout()
plt.savefig("visualizations/day2_statistical.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("   Saved → visualizations/day2_statistical.png")

# ══════════════════════════════════════════════════════════
# DAY 3 — HEATMAPS & CORRELATION
# ══════════════════════════════════════════════════════════

print("📊 Day 3: Heatmaps & Correlation...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(BG)
fig.suptitle("Heatmaps & Correlation Analysis", fontsize=18, fontweight='bold', color=TEXT)

# Correlation heatmap
corr = df[['Quantity', 'Price', 'Total_Sales']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr, annot=True, fmt='.2f', cmap='coolwarm',
    linewidths=2, linecolor=BG, square=True,
    annot_kws={"size": 14, "weight": "bold"},
    cbar_kws={"shrink": 0.8},
    ax=axes[0]
)
axes[0].set_title("Numerical Feature Correlation", fontsize=13, fontweight='bold', pad=10)

# Pivot heatmap: Region × Product → Avg Total Sales
pivot = df.pivot_table(values='Total_Sales', index='Region', columns='Product', aggfunc='mean')
sns.heatmap(
    pivot/1e3, annot=True, fmt='.0f', cmap='YlOrRd',
    linewidths=1.5, linecolor=BG,
    annot_kws={"size": 11},
    cbar_kws={"shrink": 0.8, "label": "Avg Sales (₹K)"},
    ax=axes[1]
)
axes[1].set_title("Avg Sales (₹K) — Region × Product", fontsize=13, fontweight='bold', pad=10)
axes[1].set_xlabel("Product")
axes[1].set_ylabel("Region")

plt.tight_layout()
plt.savefig("visualizations/day3_heatmaps.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("   Saved → visualizations/day3_heatmaps.png")

# ══════════════════════════════════════════════════════════
# DAY 4 — MULTI-PLOT DASHBOARD (2×2 Subplot Grid)
# ══════════════════════════════════════════════════════════

print("📊 Day 4: Multi-Plot Dashboard...")

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor(BG)
fig.suptitle("Sales Intelligence Dashboard — Static Overview", fontsize=20, fontweight='bold',
             color=TEXT, y=1.01)

# [0,0] Stacked bar: Monthly sales by Product
monthly_product = df.groupby(['Month', 'Product'])['Total_Sales'].sum().unstack(fill_value=0)
monthly_product.plot(kind='bar', stacked=True, ax=axes[0,0],
                     color=[PALETTE[p] for p in monthly_product.columns], edgecolor='none')
axes[0,0].set_title("Monthly Sales by Product", fontsize=13, fontweight='bold')
axes[0,0].set_xlabel("Month")
axes[0,0].set_ylabel("Total Sales (₹)")
axes[0,0].tick_params(axis='x', rotation=30)
axes[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x/1e6:.1f}M'))
axes[0,0].legend(title='Product', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

# [0,1] Count plot: Transactions by Region
region_counts = df['Region'].value_counts()
bars = axes[0,1].bar(region_counts.index, region_counts.values,
                     color=REGION_PALETTE, edgecolor='white', linewidth=0.5)
axes[0,1].set_title("Transaction Count by Region", fontsize=13, fontweight='bold')
axes[0,1].set_xlabel("Region")
axes[0,1].set_ylabel("Number of Transactions")
for bar, val in zip(bars, region_counts.values):
    axes[0,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                   str(val), ha='center', fontsize=11, fontweight='bold', color=TEXT)

# [1,0] Scatter: Price vs Total_Sales coloured by Product
for product, color in PALETTE.items():
    sub = df[df['Product'] == product]
    axes[1,0].scatter(sub['Price'], sub['Total_Sales']/1e3,
                      color=color, label=product, alpha=0.75, s=60, edgecolors='white', linewidths=0.3)
axes[1,0].set_title("Price vs Total Sales by Product", fontsize=13, fontweight='bold')
axes[1,0].set_xlabel("Price (₹)")
axes[1,0].set_ylabel("Total Sales (₹K)")
axes[1,0].legend(title='Product', fontsize=8)
axes[1,0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x/1e3:.0f}K'))

# [1,1] Horizontal bar: Avg Quantity by Product
avg_qty = df.groupby('Product')['Quantity'].mean().sort_values()
colors_h = [PALETTE[p] for p in avg_qty.index]
axes[1,1].barh(avg_qty.index, avg_qty.values, color=colors_h, edgecolor='white', linewidth=0.4)
axes[1,1].set_title("Avg Quantity Ordered by Product", fontsize=13, fontweight='bold')
axes[1,1].set_xlabel("Average Quantity")
for i, val in enumerate(avg_qty.values):
    axes[1,1].text(val + 0.05, i, f'{val:.1f}', va='center', fontsize=10, color=TEXT)

plt.tight_layout()
plt.savefig("visualizations/day4_multiplot.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("   Saved → visualizations/day4_multiplot.png")

# ══════════════════════════════════════════════════════════
# DAY 5 — INTERACTIVE VISUALIZATIONS WITH PLOTLY
# ══════════════════════════════════════════════════════════

print("📊 Day 5: Interactive Visualizations (Plotly)...")

PLOTLY_THEME = dict(
    template='plotly_dark',
    paper_bgcolor='#0F0F1A',
    plot_bgcolor='#1A1A2E',
    font=dict(family='Arial', color='#E8E8F0'),
)

# Interactive line chart with hover
monthly_product2 = df.groupby(['Month', 'Product'])['Total_Sales'].sum().reset_index()
fig_line = px.line(
    monthly_product2, x='Month', y='Total_Sales', color='Product',
    markers=True, title='Monthly Sales Trend by Product',
    color_discrete_map=PALETTE,
    labels={'Total_Sales': 'Total Sales (₹)', 'Month': 'Month'},
    hover_data={'Total_Sales': ':,.0f'}
)
fig_line.update_layout(**PLOTLY_THEME, title_font_size=18,
                        legend=dict(bgcolor='#1A1A2E', bordercolor='#2A2A3E'))
fig_line.update_traces(line=dict(width=2.5), marker=dict(size=8))
fig_line.write_html("visualizations/day5_interactive_line.html")

# Interactive sunburst: Region → Product
fig_sun = px.sunburst(
    df, path=['Region', 'Product'], values='Total_Sales',
    title='Sales Breakdown: Region → Product',
    color='Region', color_discrete_sequence=REGION_PALETTE,
)
fig_sun.update_layout(**PLOTLY_THEME, title_font_size=18)
fig_sun.write_html("visualizations/day5_interactive_sunburst.html")

# Interactive scatter with animation (by Month)
fig_scatter = px.scatter(
    df, x='Price', y='Total_Sales', color='Product', size='Quantity',
    animation_frame='Month', animation_group='Customer_ID',
    title='Price vs Sales Animation (Monthly)',
    color_discrete_map=PALETTE,
    labels={'Price': 'Price (₹)', 'Total_Sales': 'Total Sales (₹)', 'Quantity': 'Units'},
    size_max=30, hover_name='Customer_ID',
    hover_data={'Product': True, 'Region': True, 'Quantity': True}
)
fig_scatter.update_layout(**PLOTLY_THEME, title_font_size=18)
fig_scatter.write_html("visualizations/day5_interactive_scatter.html")

print("   Saved → visualizations/day5_interactive_*.html")

# ══════════════════════════════════════════════════════════
# DAY 6 — FULL INTERACTIVE DASHBOARD (Plotly Subplots)
# ══════════════════════════════════════════════════════════

print("📊 Day 6: Dashboard Integration...")

fig_dash = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        '📈 Monthly Sales Trend',
        '🏆 Sales Share by Product',
        '📦 Price Distribution by Product (Box)',
        '🌍 Total Sales by Region',
        '🔥 Avg Sales Heatmap (Region × Product)',
        '📊 Top Customers by Total Spend',
    ),
    specs=[
        [{"type": "xy"},    {"type": "pie"}],
        [{"type": "xy"},    {"type": "bar"}],
        [{"type": "heatmap"}, {"type": "bar"}],
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.08,
)

COLORS = list(PALETTE.values())

# [1,1] Line — monthly trend
for i, product in enumerate(df['Product'].unique()):
    sub = df[df['Product']==product].groupby('Month')['Total_Sales'].sum().reset_index()
    fig_dash.add_trace(
        go.Scatter(x=sub['Month'], y=sub['Total_Sales'], name=product,
                   line=dict(color=COLORS[i], width=2), mode='lines+markers',
                   marker=dict(size=6), legendgroup=product),
        row=1, col=1
    )

# [1,2] Pie — product share
product_share = df.groupby('Product')['Total_Sales'].sum()
fig_dash.add_trace(
    go.Pie(labels=product_share.index, values=product_share.values,
           marker_colors=COLORS, hole=0.4, textinfo='label+percent',
           hovertemplate='%{label}<br>₹%{value:,.0f}<extra></extra>',
           showlegend=False),
    row=1, col=2
)

# [2,1] Box — price by product
for i, product in enumerate(df['Product'].unique()):
    sub = df[df['Product']==product]['Price']
    fig_dash.add_trace(
        go.Box(y=sub, name=product, marker_color=COLORS[i],
               boxmean=True, legendgroup=product, showlegend=False),
        row=2, col=1
    )

# [2,2] Bar — region sales
region_sales = df.groupby('Region')['Total_Sales'].sum().sort_values(ascending=False)
fig_dash.add_trace(
    go.Bar(x=region_sales.index, y=region_sales.values,
           marker_color=REGION_PALETTE, showlegend=False,
           text=[f'₹{v/1e6:.1f}M' for v in region_sales.values],
           textposition='outside'),
    row=2, col=2
)

# [3,1] Heatmap — pivot
pivot2 = df.pivot_table(values='Total_Sales', index='Region', columns='Product', aggfunc='mean')
fig_dash.add_trace(
    go.Heatmap(
        z=pivot2.values/1e3,
        x=pivot2.columns.tolist(),
        y=pivot2.index.tolist(),
        colorscale='Viridis',
        text=np.round(pivot2.values/1e3,1),
        texttemplate='%{text}K',
        showscale=True,
        colorbar=dict(x=0.46, len=0.3, thickness=12),
        showlegend=False,
    ),
    row=3, col=1
)

# [3,2] Bar — top 10 customers
top_cust = df.groupby('Customer_ID')['Total_Sales'].sum().nlargest(10).sort_values()
fig_dash.add_trace(
    go.Bar(x=top_cust.values, y=top_cust.index, orientation='h',
           marker_color=ACCENT, showlegend=False,
           text=[f'₹{v/1e3:.0f}K' for v in top_cust.values],
           textposition='outside'),
    row=3, col=2
)

fig_dash.update_layout(
    title=dict(text='🚀  Interactive Sales Dashboard — Full Overview',
               font=dict(size=22, color='#E8E8F0'), x=0.5, xanchor='center'),
    height=1100,
    **PLOTLY_THEME,
    legend=dict(bgcolor='#1A1A2E', bordercolor='#2A2A3E', x=0.01, y=0.99),
    showlegend=True,
)

# Style all axes
for row in range(1,4):
    for col in range(1,3):
        fig_dash.update_xaxes(gridcolor='#2A2A3E', row=row, col=col)
        fig_dash.update_yaxes(gridcolor='#2A2A3E', row=row, col=col)

fig_dash.write_html("visualizations/day6_full_dashboard.html")
print("   Saved → visualizations/day6_full_dashboard.html")

# ══════════════════════════════════════════════════════════
# DAY 7 — FINAL POLISHED DASHBOARD (Static PNG for report)
# ══════════════════════════════════════════════════════════

print("📊 Day 7: Polishing final dashboard image...")

fig_final, axes = plt.subplots(2, 3, figsize=(22, 13))
fig_final.patch.set_facecolor(BG)

# Title banner
fig_final.text(0.5, 0.98, '🚀  Sales Performance Dashboard 2024',
               ha='center', va='top', fontsize=22, fontweight='bold',
               color='white', fontfamily='DejaVu Sans')
fig_final.text(0.5, 0.955, 'Powered by Seaborn + Plotly  •  100 Transactions  •  5 Products  •  4 Regions',
               ha='center', va='top', fontsize=11, color='#8888AA')

# ── Chart 1: Monthly Revenue Line ───────────────────────
ax = axes[0,0]
monthly = df.groupby('Month')['Total_Sales'].sum().reset_index()
ax.plot(range(len(monthly)), monthly['Total_Sales']/1e6,
        color=ACCENT, linewidth=2.5, marker='o', markersize=8, markerfacecolor='#F72585')
ax.fill_between(range(len(monthly)), monthly['Total_Sales']/1e6, alpha=0.18, color=ACCENT)
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly['Month'], rotation=25, ha='right', fontsize=8)
ax.set_title("Monthly Revenue Trend", fontsize=12, fontweight='bold', pad=8)
ax.set_ylabel("Revenue (₹ Millions)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x:.1f}M'))

# ── Chart 2: Product Revenue Bar ────────────────────────
ax = axes[0,1]
prod_rev = df.groupby('Product')['Total_Sales'].sum().sort_values(ascending=False)
bars = ax.bar(prod_rev.index, prod_rev.values/1e6,
              color=[PALETTE[p] for p in prod_rev.index], edgecolor='none', width=0.6)
ax.set_title("Revenue by Product", fontsize=12, fontweight='bold', pad=8)
ax.set_ylabel("Revenue (₹ Millions)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x:.1f}M'))
for bar, val in zip(bars, prod_rev.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f'₹{val/1e6:.1f}M', ha='center', fontsize=8, color=TEXT)

# ── Chart 3: Region Donut ───────────────────────────────
ax = axes[0,2]
reg_sales = df.groupby('Region')['Total_Sales'].sum()
wedges, texts, autotexts = ax.pie(
    reg_sales.values, labels=reg_sales.index,
    colors=REGION_PALETTE, autopct='%1.1f%%', startangle=140,
    pctdistance=0.78, wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2)
)
for t in texts: t.set_color(TEXT)
for at in autotexts: at.set_color('white'); at.set_fontsize(9)
ax.set_title("Revenue Share by Region", fontsize=12, fontweight='bold', pad=8)
circle = plt.Circle((0,0), 0.45, color=CARD)
ax.add_patch(circle)
total = df['Total_Sales'].sum()
ax.text(0,0, f'₹{total/1e6:.1f}M\nTotal', ha='center', va='center',
        fontsize=11, fontweight='bold', color=TEXT)

# ── Chart 4: Box Plot Price ─────────────────────────────
ax = axes[1,0]
products_order = df.groupby('Product')['Price'].median().sort_values().index.tolist()
sns.boxplot(data=df, x='Product', y='Price', order=products_order,
            palette=PALETTE, width=0.5, linewidth=1.2,
            flierprops=dict(marker='o', markerfacecolor='#F72585', markersize=4),
            ax=ax)
ax.set_title("Price Distribution by Product", fontsize=12, fontweight='bold', pad=8)
ax.set_xlabel("Product")
ax.set_ylabel("Price (₹)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{x/1e3:.0f}K'))

# ── Chart 5: Heatmap ────────────────────────────────────
ax = axes[1,1]
pivot = df.pivot_table(values='Total_Sales', index='Region', columns='Product', aggfunc='mean')
sns.heatmap(pivot/1e3, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=1.5, linecolor=BG,
            annot_kws={"size": 9, "weight": "bold"},
            cbar_kws={"shrink": 0.8, "label": "₹K"},
            ax=ax)
ax.set_title("Avg Sales (₹K) — Region × Product", fontsize=12, fontweight='bold', pad=8)

# ── Chart 6: Top 10 Customers ───────────────────────────
ax = axes[1,2]
top10 = df.groupby('Customer_ID')['Total_Sales'].sum().nlargest(10).sort_values()
bar_colors = [PALETTE[p] for p in
              [df[df['Customer_ID']==c]['Product'].mode()[0] for c in top10.index]]
ax.barh(top10.index, top10.values/1e3, color=bar_colors, edgecolor='none')
ax.set_title("Top 10 Customers by Spend", fontsize=12, fontweight='bold', pad=8)
ax.set_xlabel("Total Spend (₹K)")
for i, val in enumerate(top10.values):
    ax.text(val/1e3+1, i, f'₹{val/1e3:.0f}K', va='center', fontsize=8, color=TEXT)
legend_patches = [mpatches.Patch(color=c, label=p) for p, c in PALETTE.items()]
ax.legend(handles=legend_patches, title='Top Product', fontsize=7,
          loc='lower right', title_fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("visualizations/day7_final_dashboard.png", dpi=160, bbox_inches='tight', facecolor=BG)
plt.close()
print("   Saved → visualizations/day7_final_dashboard.png")

# ══════════════════════════════════════════════════════════
# SUMMARY STATS
# ══════════════════════════════════════════════════════════

print("\n" + "="*55)
print("  📋  KEY BUSINESS INSIGHTS")
print("="*55)
print(f"  Total Revenue       : ₹{df['Total_Sales'].sum()/1e6:.2f} Million")
print(f"  Total Transactions  : {len(df)}")
print(f"  Avg Order Value     : ₹{df['Total_Sales'].mean():,.0f}")
print(f"  Top Product         : {df.groupby('Product')['Total_Sales'].sum().idxmax()}")
print(f"  Top Region          : {df.groupby('Region')['Total_Sales'].sum().idxmax()}")
print(f"  Best Month          : {df.groupby('Month')['Total_Sales'].sum().idxmax()}")
print(f"  Highest Quantity    : {df['Quantity'].max()} units (single order)")
print("="*55)
print("\n✅ All charts generated in /visualizations/")
print("   Open visualizations/day6_full_dashboard.html for interactive view!")
