import pandas as pd
import numpy as np

# Create an expanded sales dataset
np.random.seed(42)
data_analytics = {
    'TransactionID': range(1, 101),
    'Product': np.random.choice(['Laptop', 'Mouse', 'Keyboard', 'Monitor'], 100),
    'Quantity': np.random.randint(1, 10, 100),
    'Price': np.random.randint(20, 1300, 100)
}
df_analytics = pd.DataFrame(data_analytics)
df_analytics['Revenue'] = df_analytics['Quantity'] * df_analytics['Price']

# Identify top-performing products and sales months
product_performance = df_analytics.groupby('Product')['Revenue'].sum().sort_values(ascending=False)

# What if we introduce a "Promotion" flag to see its impact?
df_analytics['Promotion'] = np.random.choice([True, False], 100, p=[0.3, 0.7])
promo_impact = df_analytics.groupby('Promotion')['Revenue'].mean()

print("--- Data Analytics ---")
print("Top 3 products by revenue:")
print(product_performance.head(3))
print("\nAverage Revenue with and without promotion:")
print(promo_impact)

if promo_impact[True] > promo_impact[False]:
    print("\nActionable Insight: Promotions seem to be effective in driving higher average revenue.")
else:
    print("\nActionable Insight: Promotions don't significantly increase revenue; consider other strategies.")