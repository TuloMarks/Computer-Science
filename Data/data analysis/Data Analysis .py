import pandas as pd

# Create a simple dataset
data = {
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Laptop', 'Monitor', 'Mouse'],
    'Quantity': [2, 5, 3, 1, 2, 4],
    'Price': [1200, 25, 75, 1250, 300, 28]
}

df_analysis = pd.DataFrame(data)

# Calculate total revenue
df_analysis['Revenue'] = df_analysis['Quantity'] * df_analysis['Price']
total_revenue = df_analysis['Revenue'].sum()

# Find the average price per product
avg_price_by_product = df_analysis.groupby('Product')['Price'].mean()

print("--- Data Analysis ---")
print("Original DataFrame:")
print(df_analysis)
print(f"\nTotal Revenue: ${total_revenue}")
print("\nAverage Price per Product:")
print(avg_price_by_product)