Remember to set $MLFLOW_TRACKING_URI
$ set -x MLFLOW_TRACKING_URI http://localhost:5000
$ echo $MLFLOW_TRACKING_URI


# Run the mlflow server and mlflow runs
$ mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1

$ mlflow run . # to just load model from file (if saved)
$ mlflow run . -e train #to train a new model
