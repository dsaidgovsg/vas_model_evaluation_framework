# This dockerfile package packages the mlflow and launch the service

FROM python:3.8

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install default-libmysqlclient-dev 

RUN pip install --upgrade pip
RUN pip install mlflow==1.7.0 mysql-connector-python mysqlclient

EXPOSE 5001

CMD LC_ALL=C.UTF-8 LANG=C.UTF-8 mlflow server --backend-store-uri mysql://127.0.0.1:3307/mlflow_training_log --default-artifact-root /artifact -p 5001
