# # Importing necessary libraries
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report

# # Load dataset
# data = load_breast_cancer()
# X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# # Initialize base models
# estimators = [('svm', SVC(probability=True)), ('dt', DecisionTreeClassifier())]

# # Meta-learner (Logistic Regression)
# stacking_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# # Fit the model
# stacking_model.fit(X_train, y_train)

# # Predictions
# y_pred = stacking_model.predict(X_test)

# # Evaluation
# print("Stacking Accuracy: ", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))


# # Importing necessary libraries
# import numpy as np
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score

# # Load dataset
# data = load_breast_cancer()
# X_train_full, X_test, y_train_full, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# # Split the training set into train and validation
# X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)

# # Base models
# model_1 = SVC(probability=True)
# model_2 = DecisionTreeClassifier()

# # Train base models on the training set
# model_1.fit(X_train, y_train)
# model_2.fit(X_train, y_train)

# # Make predictions on the validation set
# pred_1 = model_1.predict_proba(X_val)[:, 1]
# pred_2 = model_2.predict_proba(X_val)[:, 1]

# # Combine predictions as inputs for meta-model
# meta_features = np.vstack([pred_1, pred_2]).T

# # Meta-model
# meta_model = LogisticRegression()
# meta_model.fit(meta_features, y_val)

# # Test the meta-model on the test set
# test_pred_1 = model_1.predict_proba(X_test)[:, 1]
# test_pred_2 = model_2.predict_proba(X_test)[:, 1]
# test_meta_features = np.vstack([test_pred_1, test_pred_2]).T

# # Final predictions
# final_preds = meta_model.predict(test_meta_features)

# # Evaluation
# print("Blending Accuracy: ", accuracy_score(y_test, final_preds))



# Importing necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)

# Initialize RandomForest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation
print("Random Forest Accuracy: ", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
