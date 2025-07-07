# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# 2. Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['TrainFlag'] = 1
test['TrainFlag'] = 0
test['SalePrice'] = np.nan

# 3. Combine datasets
df = pd.concat([train, test], axis=0)

# 4. Handle Missing Values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna("None")
    else:
        df[col] = df[col].fillna(df[col].median())

# 5. Log-transform the target
df['SalePrice'] = np.log1p(df['SalePrice'])

# 6. Feature Engineering (basic)
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

# 7. Label Encoding (for simplicity)
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 8. Split back
train_df = df[df['TrainFlag'] == 1].drop(columns=['TrainFlag'])
test_df = df[df['TrainFlag'] == 0].drop(columns=['TrainFlag', 'SalePrice'])

X = train_df.drop(['Id', 'SalePrice'], axis=1)
y = train_df['SalePrice']

# 9. Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Ridge Regression Model
ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
ridge.fit(X_train, y_train)
val_preds = ridge.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
print("Validation RMSE (log scale):", rmse)

# 11. Predict on test set
test_preds = np.expm1(ridge.predict(test_df.drop('Id', axis=1)))

# 12. Submission
submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_preds})
submission.to_csv('submission.csv', index=False)
print("Submission saved as 'submission.csv'")
