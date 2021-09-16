FROM continuumio/miniconda3

WORKDIR /

COPY requirements.txt requirements.txt

RUN conda config --append channels conda-forge
RUN conda install --file requirements.txt

RUN apt-get update \
&& apt-get install -y --no-install-recommends git


# RUN conda update --all

# ENV MLFLOW_HOME /opt/mlflow
# ENV MLFLOW_VERSION 0.7.0
# ENV SERVER_PORT 5000
# ENV SERVER_HOST 0.0.0.0
# ENV FILE_STORE ${MLFLOW_HOME}/fileStore
# ENV ARTIFACT_STORE ${MLFLOW_HOME}/artifactStore


COPY . .
CMD [ "mlflow", "run", "."]

