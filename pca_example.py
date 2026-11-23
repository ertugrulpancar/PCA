import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

from sklearn.datasets import load_breast_cancer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
data = load_breast_cancer(as_frame=True)
df = data.frame
df.head()
df.info()
df.describe()

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

columns = data.feature_names
X_train = pd.DataFrame(X_train, columns= columns)
X_test = pd.DataFrame(X_test, columns= columns)

logistic = LogisticRegression()
gbc = GradientBoostingClassifier()

logistic.fit(X_train, y_train)
gbc.fit(X_train, y_train)

print("Logistic Regression")

y_pred_logistic = logistic.predict(X_test)
print(classification_report(y_test, y_pred_logistic))
print(confusion_matrix(y_test, y_pred_logistic))

print("Gradient Boosting Classifier")
y_pred_gbc = gbc.predict(X_test)

print(classification_report(y_test, y_pred_gbc))
print(confusion_matrix(y_test, y_pred_gbc))

pca = PCA(n_components= 4)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

X_train_pca = pd.DataFrame(X_train_pca, columns= ['PC 1', 'PC 2', 'PC 3', 'PC 4'])
X_test_pca = pd.DataFrame(X_test_pca, columns=  ['PC 1', 'PC 2', 'PC 3', 'PC 4'])



logistic = LogisticRegression()
gbc = GradientBoostingClassifier()

logistic.fit(X_train_pca, y_train)
gbc.fit(X_train_pca, y_train)

print("Logistic Regression")

y_pred_logistic = logistic.predict(X_test_pca)
print(classification_report(y_test, y_pred_logistic))
print(confusion_matrix(y_test, y_pred_logistic))

print("Gradient Boosting Classifier")
y_pred_gbc = gbc.predict(X_test_pca)

print(classification_report(y_test, y_pred_gbc))
print(confusion_matrix(y_test, y_pred_gbc))
