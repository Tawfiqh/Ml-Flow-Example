from sklearn import tree
from sklearn import datasets

import mlflow
import argparse
from sklearn import datasets

from sklearn import model_selection, linear_model


# - Testing different models
# 	- split the boston dataset into training, test AND validation
X, y = datasets.load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = model_selection.train_test_split(
    X_train, y_train, test_size=0.25
)  # 0.25 x 0.8 = 0.2


def get_flags_passed_in_from_terminal():
    parser = argparse.ArgumentParser()
    parser.add_argument("-max_depth", type=int, default=200)
    args = parser.parse_args()
    return args


ml_flow_model = True

if ml_flow_model:
    # mlflow.sklearn.autolog()

    args = get_flags_passed_in_from_terminal()
    max_depth = args.max_depth

    iteration = 0
    for max_leaf_nodes in range(2, 1000, 10):
        with mlflow.start_run() as run:
            iteration += 1
            model = tree.DecisionTreeRegressor(
                max_depth=max_depth, max_leaf_nodes=max_leaf_nodes
            )

            model.fit(X_train, y_train)
            print("Logged data and model in run {}".format(run.info.run_id))

            training_score = model.score(X_train, y_train)
            testing_score = model.score(X_test, y_test)
            validation_score = model.score(X_val, y_val)

            # mlflow.log_param("max_leaf_nodes", max_leaf_nodes)

            mlflow.log_metric("training r2 score", training_score, iteration)
            mlflow.log_metric("testing r2 score", testing_score, iteration)
            mlflow.log_metric("validation r2 score", validation_score, iteration)

            # print()
            # print("training_score", training_score)
            # print("testing_score", testing_score)
            # print("validation_score)", validation_score)
