# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load a sample dataset (the Iris dataset)
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Map target numbers to names for better understanding
target_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
iris_df['target_names'] = iris_df['target'].map(target_map)

# print the first few rows of the DataFrame
print("First 5 rows of the dataset:")
print(iris_df.head())

# Get basic information about the DataFrame
print("\nDataFrame Info:")
iris_df.info()

# Get descriptive statistics
print("\nDescriptive Statistics:")
print(iris_df.describe())

# Create some visualizations

# Scatter plot of sepal length vs sepal width, colored by species
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='target_names', data=iris_df)
plt.title('Sepal Length vs Sepal Width')
plt.show()

# Box plot of petal length for each species
plt.figure(figsize=(8, 6))
sns.boxplot(x='target_names', y='petal length (cm)', data=iris_df)
plt.title('Petal Length by Species')
plt.show()

# Histogram of petal width
plt.figure(figsize=(8, 6))
sns.histplot(data=iris_df, x='petal width (cm)', hue='target_names', kde=True)
plt.title('Distribution of Petal Width')
plt.show()