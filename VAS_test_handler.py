'''
    VAS automated performance evaluation framework
    @author: Ji Jiahao
    @date created: 20190823
    @last modified: 20190903
    @version: v0.1
    @description: Handler that generally handles tests on VAS system.
    @features: 1. mlflow used as test tracking platform
               2. generate excel test reports
'''

import os
import sys
from configparser import ConfigParser
import pandas as pd
from copy import deepcopy
import time
from metrics_evaluator import eval_metrics
from mlflow import log_metric, log_param, start_run, set_experiment

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class VAS_Test_Handler():
    def __init__(self, test_config=None):
        if not test_config:
            test_config = '/ext_vol/test_cfg.ini'

        self.config = ConfigParser(allow_no_value=True)
        self.config.read(test_config)

        os.environ["CUDA_VISIBLE_DEVICES"] = self.get_config(
            ['ENVIRONMENT', 'GPU_ID'])
        os.environ["MLFLOW_TRACKING_URI"] = self.get_config(
            ['ENVIRONMENT', 'MLFLOW_TRACKING_URI'])

        # self.test_dir = os.getcwd()
        self.vas_software_path = self.get_config(
            ['VAS SOFTWARE', 'PATH'])
        # sys.path.append(self.test_dir)
        sys.path.append(self.vas_software_path)

    def run(self):
        test_list = self.get_test_list(self.get_data_summary_file_path())

        '''change working dir'''
        os.chdir(self.vas_software_path)

        # DEBUG ##
        # from pprint import pprint
        # pprint(test_list)
        # sys.exit()
        # ########

        _ = self.conduct_test(test_list, self.vas_software_path)
        # self.__test_mlflow()

    def get_test_list(self, data_summary_path):
        '''
            return type: list of dictionaries. One test case per dictionary.
                         dictionary keys are excel column names.
        '''
        data_summary = pd.read_excel(data_summary_path)
        test_list = []
        column_names = list(data_summary.columns)

        for row in data_summary.itertuples():
            test_case = {}
            row_items = list(row)
            for i in range(len(column_names)):
                test_case.update({column_names[i]: row_items[i + 1]})
            test_list.append(deepcopy(test_case))

        return test_list

    def __test_mlflow(self):
        exp_name = self.get_config(['EXPERIMENT METADATA', 'EXP_NAME'])
        run_name = self.get_config(['EXPERIMENT METADATA', 'RUN_NAME'])
        set_experiment(exp_name)
        with start_run(run_name=run_name):
            log_param('dummy_param', 1)
            log_metric('dummy_metric', 2)

    def conduct_test(self, test_list, vas_path):
        '''
            Model prediction interface
            - init_vas_model() - initialize model once and return the active session
            - call_vas_inference() - run model on test data stream and return results
            - add mlflow log
        '''

        from predict import Predictor
        predictor = Predictor(vas_path)
        # model_init_start = time.time()
        predictor.model_init()
        # log_metric("model_init_time", time.time() - model_init_start)

        for test in test_list:
            exp_name = self.get_config(['EXPERIMENT METADATA', 'EXP_NAME'])
            run_name = self.get_config(['EXPERIMENT METADATA', 'RUN_NAME'])
            set_experiment(exp_name)
            with start_run(run_name=run_name):
                result = predictor.model_infer(
                    self.get_file_path(test['File_name']))

                test.update(result)

                # mlflow log
                for k, v in test.items():
                    if k.startswith('meta_'):
                        log_param(k[5:], v)
                    else:
                        log_param(k, v)

                metrics = eval_metrics(test_result=test, metrics_to_eval=[
                    'explained_variance_score'])

                for k, v in metrics.items():
                    log_metric(k, v)

        return test_list

    def get_config(self, config_keys):
        result = self.config
        for key in config_keys:
            result = result[key]
        return result

    def get_data_summary_file_path(self):
        summary_file_dir = self.get_config(
            ['TEST DATA', 'DATA_SUMMARY_EXCEL_PATH'])
        if not summary_file_dir.startswith('/'):
            # relative path
            if summary_file_dir.startswith('./'):
                summary_file_dir = summary_file_dir[2:]
            # test_file_dir = self.test_dir + '/' + test_file_dir
            summary_file_dir = '/ext_vol/' + summary_file_dir

        return summary_file_dir

    def get_file_path(self, file_name):
        test_file_dir = self.get_config(['TEST DATA', 'TEST_DATA_PATH'])
        if not test_file_dir.startswith('/'):
            # relative path
            if test_file_dir.startswith('./'):
                test_file_dir = test_file_dir[2:]
            # test_file_dir = self.test_dir + '/' + test_file_dir
            test_file_dir = '/ext_vol/' + test_file_dir
        file_path = os.path.join(
            test_file_dir, file_name)
        return file_path


if __name__ == '__main__':
    th = VAS_Test_Handler()
    th.run()
