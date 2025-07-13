import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv("C:\Users\Ishmeet\OneDrive\Documents\GitHub\IshmeetKaur222-Celebal-Internship-Codes\winequality-red.csv")


df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

# Features and labels
X = df.drop("quality", axis=1)
y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB()
}

# Evaluation
print("=== Initial Model Evaluation ===\n")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

# Hyperparameter Tuning: Random Forest (Grid Search)
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(RandomForestClassifier(), rf_params, cv=5, scoring='f1', n_jobs=-1)
grid_rf.fit(X_train, y_train)
print("\n=== GridSearchCV - Random Forest ===")
print("Best Params:", grid_rf.best_params_)
print("Best F1 Score (Train CV):", grid_rf.best_score_)

# Hyperparameter Tuning: SVM (Randomized Search)
svm_params = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
random_svc = RandomizedSearchCV(SVC(), svm_params, cv=5, scoring='f1', n_iter=10, n_jobs=-1, random_state=42)
random_svc.fit(X_train, y_train)
print("\n=== RandomizedSearchCV - SVM ===")
print("Best Params:", random_svc.best_params_)
print("Best F1 Score (Train CV):", random_svc.best_score_)

# Final evaluation on test set
best_model = grid_rf.best_estimator_
y_best_pred = best_model.predict(X_test)
print("\n=== Final Best Model Evaluation (Random Forest) ===")
print(classification_report(y_test, y_best_pred))
