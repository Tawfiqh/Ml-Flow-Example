from sklearn import tree
from sklearn import datasets
from sklearn.metrics import r2_score

import mlflow.pyfunc
import mlflow
import argparse
from sklearn import datasets

from sklearn import model_selection


# - Testing different models
# 	- split the boston dataset into training, test AND validation
X, y = datasets.load_boston(return_X_y=True)


model_name = "sklearn-DecisionTree-model"
model_version = 10
model_uri = f"models:/{model_name}/{model_version}"
print(f"model_uri: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri=model_uri)

y_hat = model.predict(X)
print("y_hat", y_hat)
print("y_hat-y", y_hat - y)
test_score = r2_score(y, y_hat)
print(f"test_score: {test_score}")
