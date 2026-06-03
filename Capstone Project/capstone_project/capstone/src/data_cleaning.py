"""
data_cleaning.py
Utility module: loads raw CSVs, cleans them, saves cleaned versions.
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def clean_churn(path_in, path_out):
    df = pd.read_csv(path_in)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df['Churn_Label'] = df['Churn'].map({1: 'Yes', 0: 'No'})
    df['Tenure_Bin'] = pd.cut(df['Tenure'], bins=[0,12,24,48,72],
                               labels=['0-1yr','1-2yr','2-4yr','4-6yr'])
    df.to_csv(path_out, index=False)
    return df

def clean_houses(path_in, path_out):
    df = pd.read_csv(path_in)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df['Price_Lakhs'] = df['Price'] / 100000
    df['Price_Per_Sqft'] = df['Price'] / df['Area']
    df['Age_Bin'] = pd.cut(df['Age'], bins=[-1,5,15,30,50],
                            labels=['New','Young','Mid','Old'])
    df.to_csv(path_out, index=False)
    return df

def clean_sales(path_in, path_out):
    df = pd.read_csv(path_in)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month_name()
    df['Quarter'] = df['Date'].dt.to_period('Q').astype(str)
    df['Margin'] = ((df['Total_Sales'] - df['Quantity'] * df['Price'] * 0.6)
                    / df['Total_Sales'] * 100).round(2)
    df.drop_duplicates(inplace=True)
    df.to_csv(path_out, index=False)
    return df

if __name__ == '__main__':
    clean_churn(f'{DATA_DIR}/customer_churn_raw.csv', f'{DATA_DIR}/customer_churn_clean.csv')
    clean_houses(f'{DATA_DIR}/house_prices_raw.csv', f'{DATA_DIR}/house_prices_clean.csv')
    clean_sales(f'{DATA_DIR}/sales_data_raw.csv', f'{DATA_DIR}/sales_data_clean.csv')
    print("All datasets cleaned and saved.")
