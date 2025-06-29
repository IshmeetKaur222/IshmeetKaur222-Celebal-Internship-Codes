# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset 
df = sns.load_dataset('titanic')

# Set style
sns.set(style="whitegrid")

# 1. Basic Information

print("First 5 rows:\n", df.head())
print("\nInfo:\n")
print(df.info())
print("\nSummary Statistics:\n", df.describe(include='all'))

# 2. Missing Values

print("\nMissing Values:\n", df.isnull().sum())

# Drop high-missing columns
df.drop(columns=['deck', 'embark_town', 'alive'], inplace=True)

# Fill missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['fare'].fillna(df['fare'].mean(), inplace=True)

# 3. Visualizing Distributions

# Histograms
df[['age', 'fare']].hist(bins=30, figsize=(12, 4), edgecolor='black', color='skyblue')
plt.suptitle('Age and Fare Distributions')
plt.show()

# Countplot for Survived by Sex
sns.countplot(x='survived', hue='sex', data=df)
plt.title('Survival Count by Gender')
plt.show()

# 4. Outlier Detection

plt.figure(figsize=(10, 6))
sns.boxplot(data=df[['age', 'fare', 'sibsp', 'parch']])
plt.title('Boxplot for Outlier Detection')
plt.show()

# 5. Correlation Heatmap

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 6. Survival by Categorical Features

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(x='pclass', hue='survived', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Survival by Pclass')

sns.countplot(x='sex', hue='survived', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Survival by Sex')

sns.countplot(x='embarked', hue='survived', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Survival by Embarkation')

sns.histplot(x='fare', hue='survived', data=df, bins=30, kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Fare Distribution by Survival')

plt.tight_layout()
plt.show()

# 7. Feature Engineering: Family Size

df['family_size'] = df['sibsp'] + df['parch'] + 1

sns.boxplot(x='survived', y='family_size', data=df)
plt.title('Family Size vs Survival')
plt.show()

# 8. Pairplot for Relationship Exploration

sns.pairplot(df[['age', 'fare', 'pclass', 'survived']], hue='survived', palette='Set1')
plt.suptitle("Pairplot of Numeric Features Colored by Survival", y=1.02)
plt.show()
