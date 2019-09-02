# vas_model_evaluation_framework
VAS automated performance evaluation framework

(the documentation under development)

Features 
    • Automated VAS system experiment evaluation and logging. Plug in model, test data and evaluate.
    • Mlflow with MySQL backend, which supports multiple evaluation task running at the same time.

Prepare VAS Software and Test Data
(Refer to sample script)
    1. Prepare predict.py
    2. Prepare test data and data summary excel
    3. Prepare test configuration test_cfg.ini

Launch Test
    1. launch mysql docker service if it is not up. Run Command 1
    2. launch cuda environment on dgx. Run Command 2
    3. In the docker container, launch mlflow server in the background.  Run Command 3
    4. In the docker container, launch the test script. Run Command 4

Visualize and Save Results
    1. logon to dgx through vnc, go localhost:5001 in browser.
    2. export the experiment data from mlflow web interface. 
    3. Collect artifact from /var/experiment_data/artifact

MySQL docker side note:
    • When launching mysql container, map the data storage to persistent local storage. So when the container is down, the experiment data won’t be lost. 
    • Backup /var/experiment_data/mysql frequently.

Command List
    1. docker run --name mlflow_sql_backend --restart always -p 3306:3306 -e MYSQL_ALLOW_EMPTY_PASSWORD=1 -d -v /var/experiment_data/mysql:/var/lib/mysql localhost:5000/mysql
    2. nvidia-docker run -it --rm -v ~/vol_to_docker_img:/ext_vol --network=host -e LC_ALL=C.UTF-8 -e LANG=C.UTF-8 localhost:5000/cuda_env:0828
    3. mlflow server --backend-store-uri mysql://127.0.0.1:3306/mlflow_experiments --default-artifact-root /ext_vol/mlflow_artifact -p 5001 &
    4. cd /ext_vol/vas_test; python3 VAS_test_handler.py

TODO:
    1. design issue. clean up params log, maybe save metadata as artifact
    2. design issue. test result hierachy.
        1. Project – experiment – average metrics
        2. Project_experiment – test case – detailed metrics 
