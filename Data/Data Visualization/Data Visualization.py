import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reuse the dataframe from Data Analysis
data = {
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
    'Average Price': [1225.0, 26.5, 75.0, 300.0] # These are from the previous analysis
}
df_viz = pd.DataFrame(data)

# Create a bar chart of average prices
plt.figure(figsize=(8, 5))
sns.barplot(x='Product', y='Average Price', data=df_viz)
plt.title('Average Price by Product')
plt.xlabel('Product')
plt.ylabel('Average Price ($)')
plt.show()

# Create a pie chart to show the proportion of average prices
plt.figure(figsize=(8, 8))
plt.pie(df_viz['Average Price'], labels=df_viz['Product'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Distribution of Average Product Prices')
plt.show()