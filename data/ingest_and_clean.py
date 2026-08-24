import os
import pandas as pd
from datasets import load_dataset
import sqlite3
import numpy as np

def clean_data(df):
    print("Initial columns:", df.columns)
    
    # Rename specifically known columns to our target schema
    rename_mapping = {
        'approx_cost(for two people)': 'cost',
        'rate': 'rating',
        'cuisines': 'cuisine'
    }
    df = df.rename(columns=rename_mapping)
    print("Columns after renaming:", df.columns)
    
    # Drop rows with missing critical values
    critical_cols = [c for c in ['name', 'location', 'cuisine', 'cost', 'rating'] if c in df.columns]
    df = df.dropna(subset=critical_cols)
    
    # Standardize text
    if 'location' in df.columns:
        df['location'] = df['location'].astype(str).str.lower().str.strip()
    if 'cuisine' in df.columns:
        df['cuisine'] = df['cuisine'].astype(str).str.lower().str.strip()
        
    # Map numerical costs to budget tiers (Low, Medium, High)
    if 'cost' in df.columns:
        # Remove commas and convert to float
        df['cost'] = df['cost'].astype(str).str.replace(',', '').str.strip()
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        
        # Let's say < 500 is Low, 500-1500 is Medium, > 1500 is High
        def categorize_budget(cost):
            if pd.isna(cost):
                return 'unknown'
            elif cost < 500:
                return 'low'
            elif cost <= 1500:
                return 'medium'
            else:
                return 'high'
        
        df['budget'] = df['cost'].apply(categorize_budget)
        
    if 'rating' in df.columns:
        # Rating is like '4.1/5' or 'NEW' or '-'
        df['rating'] = df['rating'].astype(str).str.split('/').str[0].str.strip()
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        # Drop rows where rating could not be parsed
        df = df.dropna(subset=['rating'])
        
    return df

def main():
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("ManikaSaini/zomato-restaurant-recommendation", split="train")
    df = dataset.to_pandas()
    
    print(f"Dataset loaded with {len(df)} rows.")
    
    print("Cleaning data...")
    df_clean = clean_data(df)
    
    print(f"Data cleaned, {len(df_clean)} rows remaining.")
    
    # Save as CSV
    csv_path = os.path.join(os.path.dirname(__file__), "restaurants.csv")
    df_clean.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")
    
    # Save to SQLite DuckDB / DB
    db_path = os.path.join(os.path.dirname(__file__), "restaurants.db")
    conn = sqlite3.connect(db_path)
    df_clean.to_sql("restaurants", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved to database at {db_path}")

if __name__ == "__main__":
    main()
