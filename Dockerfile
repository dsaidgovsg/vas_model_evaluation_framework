# This dockerfile packages the source VAS model evaluator with the environment

# base environment containing only the dependencies
# FROM localhost:5000/vas_test_framework:base_env
FROM localhost:5000/vas_test_framework:temp_for_mobius

# copy latest test handler
COPY VAS_test_handler.py /vas_test/
COPY metrics_evaluator.py /vas_test/

# launch mlflow
# CMD mlflow server \
#	--backend-store-uri mysql://127.0.0.1:3306/mlflow_experiments \
#	--default-artifact-root /ext_vol/mlflow_artifact \
#	-p 5001 & \
# && EXPOSE 5001

# launch test
WORKDIR /vas_test/
CMD python3 VAS_test_handler.py -d 0
