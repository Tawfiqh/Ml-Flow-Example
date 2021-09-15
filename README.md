$ set -x MLFLOW_TRACKING_URI http://localhost:5000

$ echo $MLFLOW_TRACKING_URI

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1
