Remember to set $MLFLOW_TRACKING_URI
$ set -x MLFLOW_TRACKING_URI http://localhost:5000
$ echo $MLFLOW_TRACKING_URI


# Run the mlflow server and mlflow runs
$ mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1

$ mlflow run .
$ mlflow run . -P load_model=1  # to just load model from file (if saved)