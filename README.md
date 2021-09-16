Remember to set $MLFLOW_TRACKING_URI
$ set -x MLFLOW_TRACKING_URI http://localhost:5000
$ echo $MLFLOW_TRACKING_URI


# Run the mlflow server and mlflow runs
$ mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1

$ mlflow run . # to just load model from file (if saved)
$ mlflow run . -e train #to train a new model

# Docker
Build the Dockerfile into an image (and then we run the image as a container)

$ docker build --tag aicore-mlflow-docker . 
$ docker run -p 8000:8000 --name mlflow-fast-api-test aicore-mlflow-docker 

Run with -d to run it in the background (detached)
$ docker run -p 8000:8000 --name mlflow-fast-api-test aicore-mlflow-docker 

$ docker start mlflow-fast-api-test
$ docker stop mlflow-fast-api-test

Now can ping the API on 0.0.0.0:8000

docker run -i -t conda/miniconda3 /bin/bash

Can we do miniconda3- with a lower version of python

or a lower version of any of our other packages
