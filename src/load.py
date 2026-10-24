import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def to_number(num):
    if not isinstance(num, str) or pd.isna(num):
        return num
    clean_val = num.replace(".", "").replace(",", ".")
    try:
        return float(clean_val)
    except ValueError:
        return np.nan

def load_and_clean_data(generation_path=BASE_DIR / 'data' / 'raw' / 'generation.csv', 
    prices_path=BASE_DIR / 'data' / 'raw' / 'prices.csv'):
    # Load raw data
    generation = pd.read_csv(
        generation_path, 
        sep=';', decimal=',', thousands='.', 
        na_values=['-', 'n.a.', '']
    )
    prices = pd.read_csv(
        prices_path, 
        sep=';', decimal=',', thousands='.', 
        na_values=['-', 'n.a.', '']
    )

    # Format datetime columns
    generation['Start date'] = pd.to_datetime(generation['Start date'])
    generation['End date'] = pd.to_datetime(generation['End date'])
    prices['Start date'] = pd.to_datetime(prices['Start date'])
    prices['End date'] = pd.to_datetime(prices['End date'])

    generation.to_parquet('cleaned_generation.parquet')
    prices.to_parquet('cleaned_prices.parquet')

    # Assemble target DataFrame
    df = pd.DataFrame(index=prices.index)
    df['price_eur_mwh'] = prices['Germany/Luxembourg [€/MWh] Calculated resolutions']
    df['wind_onshore_mwh'] = generation['Wind onshore [MWh] Calculated resolutions']
    df['wind_offshore_mwh'] = generation['Wind offshore [MWh] Calculated resolutions']
    df['solar_mwh'] = generation['Photovoltaics [MWh] Calculated resolutions']

    # Remove rows with missing price targets
    df = df.dropna(subset=['price_eur_mwh'], axis=0)

    # Apply float conversion to all numeric columns
    vectorized_cleaner = np.vectorize(to_number, otypes=[float])
    df = pd.DataFrame(vectorized_cleaner(df), columns=df.columns, index=df.index)

    # Calculate total variable renewables (Executed after float conversion to prevent string concatenation)
    df['variable_renewables_mwh'] = df['wind_onshore_mwh'] + df['wind_offshore_mwh'] + df['solar_mwh']

    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    
    # Summary statistics
    neg_hours = (df["price_eur_mwh"] < 0).sum()
    total_hours = len(df)
    print(f"Total hours with negative prices: {neg_hours} out of {total_hours} ({neg_hours/total_hours:.1%})")

    # Plot Distribution
    fig = plt.figure(dpi=100, figsize=(8, 4))
    sns.histplot(data=df, x='price_eur_mwh', bins=100, kde=True, color='royalblue')
    plt.xlabel('Price (€/MWh)')
    plt.ylabel('Frequency (Hours)')
    plt.title('Distribution of Day-Ahead Electricity Prices in Germany')
    plt.tight_layout()
    plt.show()

