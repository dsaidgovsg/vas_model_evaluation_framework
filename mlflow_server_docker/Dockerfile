FROM ubuntu:16.04

RUN apt-get -y update && apt-get -y upgrade 
RUN apt-get -y install python-pip python3-dev libmysqlclient-dev

RUN pip install --upgrade pip
RUN pip install mlflow mysql-connector-python mysqlclient

CMD set -e
CMD mlflow server --backend-store-uri mysql://127.0.0.1:3306/mlflow_experiments --default-artifact-root /artifact -p 5001 
CMD tail -f /dev/null
