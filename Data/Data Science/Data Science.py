import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# A data scientist's project: predict customer spending.

# 1. Data Collection & Preprocessing
# (Using our customer data from before)
df_science = pd.DataFrame({
    'AnnualIncome': np.random.randint(30000, 150000, 500),
    'Age': np.random.randint(18, 65, 500),
    'SpendingScore': np.random.randint(1, 100, 500)
})
# Engineer a relationship for demonstration
df_science['SpendingScore'] = df_science['SpendingScore'] + (df_science['AnnualIncome'] / 2000)

# 2. Exploratory Data Analysis (EDA) & Visualization
# (Analyze the data and visualize relationships)
print("--- Data Science Workflow ---")
print("Step 1: EDA - Checking correlations")
print(df_science.corr())
sns.pairplot(df_science)
plt.suptitle('Pairplot of Customer Data', y=1.02)
plt.show()

# 3. Machine Learning (Predictive Modeling)
# (Build a model to predict spending score)
X = df_science[['AnnualIncome', 'Age']]
y = df_science['SpendingScore']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f"\nStep 2: Predictive Modeling")
print(f"Mean Squared Error of the model: {mse:.2f}")

# 4. Data Analytics (Interpretation & Action)
# (Use the model to make predictions and business recommendations)
new_customer = pd.DataFrame([{'AnnualIncome': 100000, 'Age': 45}])
predicted_spending = model.predict(new_customer)

print(f"\nStep 3: Actionable Insights")
print(f"Predicted spending for a new customer with $100k income and 45 years old: {predicted_spending[0]:.2f}")
print("\nFinal Recommendation: The model can be used to identify potential high-value customers.")