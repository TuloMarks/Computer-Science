import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set a random seed for reproducibility
np.random.seed(42)

# Create a synthetic dataset
n_customers = 1000

data = {
    'CustomerID': range(1, n_customers + 1),
    'Age': np.random.randint(18, 70, n_customers),
    'AnnualIncome': np.random.randint(30000, 150000, n_customers),
    'SpendingScore': np.random.randint(1, 100, n_customers),
    'LoyaltyMonths': np.random.randint(1, 60, n_customers),
    'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_customers)
}

df = pd.DataFrame(data)

# Introduce a relationship for demonstration (e.g., higher income leads to higher spending)
df['SpendingScore'] = np.clip(df['SpendingScore'] + (df['AnnualIncome'] - 90000) / 2000, 1, 100).astype(int)

print("Generated Dataset Head:")
print(df.head())
print("\n" + "="*50 + "\n")

# --- 1. Data Analysis & Data Analytics ---
# Data Analysis: Answering a specific question
print("--- 1. Data Analysis ---")
avg_spending_by_city = df.groupby('City')['SpendingScore'].mean().sort_values(ascending=False)
print("Average Spending Score by City:")
print(avg_spending_by_city)

# Data Analytics: Broader insights and strategic thinking
print("\n--- 2. Data Analytics ---")
# Let's define a "high-value customer" and analyze their characteristics
high_value_customers = df[(df['SpendingScore'] > 75) & (df['AnnualIncome'] > 100000)]
print(f"Number of high-value customers: {len(high_value_customers)}")
print("Average age of high-value customers:", high_value_customers['Age'].mean())
print("\n" + "="*50 + "\n")


# --- 2. Data Visualization ---
print("--- 3. Data Visualization ---")
# Visualize the relationship between income and spending
plt.figure(figsize=(10, 6))
sns.scatterplot(x='AnnualIncome', y='SpendingScore', data=df, hue='City')
plt.title('Annual Income vs. Spending Score')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='City')
plt.show()

# Visualize spending distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['SpendingScore'], bins=20, kde=True)
plt.title('Distribution of Spending Scores')
plt.xlabel('Spending Score')
plt.ylabel('Frequency')
plt.show()
print("\n" + "="*50 + "\n")


# --- 3. Data Mining & Machine Learning ---
print("--- 4. Data Mining & Machine Learning ---")
# Data Mining: Using K-Means to find customer segments (clustering)
# We don't have a pre-defined label, we are "mining" for patterns
features_for_clustering = df[['AnnualIncome', 'SpendingScore']]

# Standardize the features for better clustering performance
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_for_clustering)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

print("Customer segments found via K-Means clustering:")
print(df.groupby('Cluster').agg({
    'CustomerID': 'count',
    'AnnualIncome': 'mean',
    'SpendingScore': 'mean'
}))

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='AnnualIncome', y='SpendingScore', hue='Cluster', data=df, palette='viridis')
plt.title('Customer Segments (K-Means Clustering)')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# Machine Learning: Supervised learning example (predicting spending score)
# This is a regression task
print("\n--- 5. Machine Learning (Predictive Modeling) ---")
X = df[['AnnualIncome', 'Age', 'LoyaltyMonths']]
y = df['SpendingScore']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error of the model: {mse:.2f}")

# Let's predict the spending score for a new customer
new_customer = pd.DataFrame([[110000, 35, 24]], columns=['AnnualIncome', 'Age', 'LoyaltyMonths'])
predicted_spending = model.predict(new_customer)
print(f"Predicted Spending Score for new customer: {predicted_spending[0]:.2f}")
print("\n" + "="*50 + "\n")


# --- 4. Data Science ---
print("--- 6. Data Science (Putting it all together) ---")
# Data Science is the overarching process. A data scientist would:
# 1. Start with a business question (e.g., "How can we increase customer spending?")
# 2. Perform Data Analysis to understand current trends.
# 3. Use Data Visualization to present findings.
# 4. Use Data Mining (K-Means) to discover customer segments.
# 5. Use Machine Learning (Linear Regression) to predict spending scores for new customers.
# 6. Finally, use these insights to provide a complete, data-driven recommendation to the business.
print("The script has demonstrated a simplified data science workflow, from data generation to predictive modeling.")