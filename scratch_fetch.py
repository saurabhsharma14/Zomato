from datasets import load_dataset
import pandas as pd

dataset = load_dataset("ManikaSaini/zomato-restaurant-recommendation", split="train")
needed_cols = ['name', 'location', 'cuisines', 'approx_cost(for two people)', 'rate']
dataset = dataset.select_columns([c for c in needed_cols if c in dataset.column_names])
df = dataset.to_pandas()

rename_mapping = {
    'approx_cost(for two people)': 'cost',
    'rate': 'rating',
    'cuisines': 'cuisine'
}
df = df.rename(columns=rename_mapping)

critical_cols = [c for c in ['name', 'location', 'cuisine', 'cost', 'rating'] if c in df.columns]
df = df.dropna(subset=critical_cols)

if 'location' in df.columns:
    df['location'] = df['location'].astype(str).str.lower().str.strip()
if 'cuisine' in df.columns:
    df['cuisine'] = df['cuisine'].astype(str).str.lower().str.strip()

if 'cost' in df.columns:
    df['cost'] = df['cost'].astype(str).str.replace(',', '').str.strip()
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')

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

df.to_parquet('data/cleaned_restaurants.parquet', index=False)
print("Saved to data/cleaned_restaurants.parquet")
