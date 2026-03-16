import pandas as pd
import numpy as np

def generate_data():
    np.random.seed(42)
    # Creating 2000 synthetic transactions
    data = {
        'distance_from_home': np.random.exponential(scale=10, size=2000),
        'purchase_price_ratio': np.random.normal(loc=1.5, scale=0.5, size=2000),
        'online_order': np.random.choice([0, 1], size=2000, p=[0.4, 0.6]),
        'fraud': np.random.choice([0, 1], size=2000, p=[0.94, 0.06])
    }
    pd.DataFrame(data).to_csv('transactions.csv', index=False)
    print("✅ Created transactions.csv!")

if __name__ == "__main__":
    generate_data()