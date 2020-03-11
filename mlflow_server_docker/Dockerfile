# This dockerfile package packages the mlflow and launch the service

FROM ubuntu:16.04

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get -y install python3-pip python3-dev libmysqlclient-dev

RUN pip3 install --upgrade pip
RUN pip3 install mlflow==1.7.0 mysql-connector-python mysqlclient

EXPOSE 5001

CMD LC_ALL=C.UTF-8 LANG=C.UTF-8 mlflow server --backend-store-uri mysql://127.0.0.1:3306/mlflow_training_log --default-artifact-root /artifact -p 5001
