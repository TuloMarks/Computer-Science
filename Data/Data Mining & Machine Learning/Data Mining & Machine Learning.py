import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate customer data for clustering
customer_data = {
    'CustomerID': range(1, 201),
    'AnnualIncome': np.random.randint(30000, 150000, 200),
    'SpendingScore': np.random.randint(1, 100, 200)
}
df_ml = pd.DataFrame(customer_data)

# Introduce a pattern for the algorithm to "find"
df_ml.loc[df_ml['AnnualIncome'] > 120000, 'SpendingScore'] = df_ml.loc[df_ml['AnnualIncome'] > 120000, 'SpendingScore'] + 20

# Scale the data for the algorithm
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df_ml[['AnnualIncome', 'SpendingScore']])

# Data Mining with K-Means (Unsupervised Learning)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_ml['Cluster'] = kmeans.fit_predict(features_scaled)

print("--- Data Mining & Machine Learning ---")
print("Customer segmentation (K-Means Clustering):")
print(df_ml.groupby('Cluster').agg({
    'CustomerID': 'count',
    'AnnualIncome': 'mean',
    'SpendingScore': 'mean'
}))

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='AnnualIncome', y='SpendingScore', hue='Cluster', data=df_ml, palette='viridis', style='Cluster', s=100)
plt.title('Customer Segments Discovered with K-Means')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.show()