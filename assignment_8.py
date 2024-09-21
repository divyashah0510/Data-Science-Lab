# # Importing necessary libraries
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report

# # Load dataset (Breast Cancer Dataset)
# data = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# # Initialize AdaBoost model
# ada_model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)

# # Fit the model
# ada_model.fit(X_train, y_train)

# # Predictions
# y_pred = ada_model.predict(X_test)

# # Evaluation
# print("AdaBoost Accuracy: ", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # Importing necessary libraries
# from catboost import CatBoostClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
# from sklearn.metrics import accuracy_score, classification_report

# # Load dataset (Breast Cancer Dataset)
# data = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# # Initialize CatBoost model
# cat_model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=5, verbose=0)

# # Fit the model
# cat_model.fit(X_train, y_train)

# # Predictions
# y_pred = cat_model.predict(X_test)

# # Evaluation
# print("CatBoost Accuracy: ", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# Import necessary libraries
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# Initialize XGBoost model
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1)

# Fit the model
xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Evaluation
print("XGBoost Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
