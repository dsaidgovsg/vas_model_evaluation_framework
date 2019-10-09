# VAS Model Evaluation Framework
VAS automated performance evaluation framework v0.1.1 (dev)

## Features 
 - Automated and centralized VAS system experiment evaluation and logging platform. Simply plug in model, test data and config file. Then evaluate.
 - Mlflow with MySQL backend, which supports multiple evaluation task running at the same time.
 
## Architecture
<img src="https://github.com/dsaidgovsg/vas_model_evaluation_framework/blob/master/architecture_v0.1.jpg" width="300">

## Test Environment Setup (this has been done on dgx)
1. Launch mysql docker service if it is not up. Run `docker run --name mlflow_sql_backend --restart always -p 3306:3306 -e MYSQL_ALLOW_EMPTY_PASSWORD=1 -d -v /var/experiment_data/mysql:/var/lib/mysql localhost:5000/mysql` to launch the DB server.
2. Build mlflow server image and launch. Run `docker build -t localhost:5000/mlflow-server ./mlflow_server_docker/` to build. Run `docker run -d --name mlflow_server --restart always --net=host -v /var/experiment_data/artifact:/artifact localhost:5000/mlflow-server` to launch the mlflow server.
2. Build Docker Image. Run `docker build -t localhost:5000/vas_test_framework:v0.1 .` on dgx to build docker image for the VAS model evaluation framework

## Prepare VAS Software and Test Data
(Refer to sample experiment folder)
 - Prepare software with trained model and predict.py 
 - Prepare test data and data summary excel
 - Prepare test configuration test_cfg.ini

## Launch Test
Run `nvidia-docker run --rm --network=host -v <path_to_experiment_folder>:/ext_vol localhost:5000/vas_test_framework:v0.1` on dgx

## Visualize and Save Results
 1. logon to dgx through vnc, go to localhost:5001 in browser.
 2. export the experiment data from mlflow web interface. 
 3. Collect artifact from /var/experiment_data/artifact

## Version history

### v0.1.1 
 1. added new evaluation metrics: mAP.
 2. redefine data summary columns.

### v0.1
 1. first stable version

### MySQL docker side note:
 - When launching mysql container, map the data storage to persistent local storage. So when the container is down, the experiment data won’t be lost. 
 - Backup /var/experiment_data/mysql frequently.

### TODO:
 1. support more evaluation metrics. 
 2. allow to define customized metrics.
 3. auto generate test report.
 4. include support for artifact logging.
