import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset 🌸
# The Iris dataset is built into scikit-learn for easy access.
iris = load_iris(as_frame=True)
X = iris.data  # Features (sepal length, sepal width, etc.)
y = iris.target  # The target variable (species of iris)

# Let's peek at the data and its structure
print("--- Iris Dataset Snapshot ---")
print("Features (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())

# The target is an integer (0, 1, 2) that corresponds to a species name.
# We can look up the species names from the dataset.
print("\nTarget names:", iris.target_names)
print("\n" + "="*50 + "\n")

# 2. Split the data into training and testing sets
# We need to train our model on a portion of the data and then test it on unseen data.
# This helps us evaluate how well the model generalizes.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- Data Splitting ---")
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print("\n" + "="*50 + "\n")

# 3. Train a machine learning model
# We'll use a Decision Tree Classifier for this example.
# We 'fit' the model to the training data, which is how it "learns."
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. Make predictions and evaluate the model
# Now we use our trained model to predict the species for the test set.
predictions = model.predict(X_test)

# We can compare these predictions to the actual species (y_test).
accuracy = accuracy_score(y_test, predictions)

print("--- Model Evaluation ---")
print("First 10 predictions on test data:", predictions[:10])
print("First 10 actual labels:", y_test.values[:10])
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")