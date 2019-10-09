'''
    VAS automated performance evaluation framework
    @author: Ji Jiahao
    @date created: 20190823
    @last modified: 20190911
    @version: v0.1.1
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
from metrics_evaluator import eval_metrics_video, eval_metrics_image_bb
from mlflow import log_metric, log_param, start_run, set_experiment
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DEBUG = True


class VAS_Test_Handler():
    def __init__(self, test_config=None):
        if not test_config:
            if DEBUG:
                test_config = '/ext_vol/vas_model_evaluation_framework/sample_experiment/test_cfg.ini'
            else:
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
        test_list = self.get_test_list(
            self.get_path('DATA_SUMMARY_EXCEL_PATH'))

        ''' IMPT: change working dir'''
        os.chdir(self.vas_software_path)

        predictor = self.init_test()

        if self.get_config(['TEST DATA', 'TEST_DATA_TYPE']).lower() == 'video':
            _ = self.conduct_test_video(test_list, predictor)
        elif self.get_config(['TEST DATA', 'TEST_DATA_TYPE']).lower() == 'image':
            _ = self.conduct_test_image(test_list, predictor)

        # self.__test_mlflow()

    def init_test(self):
        '''
            - init_vas_model() - initialize model once and return the active session
        '''
        from predict import Predictor
        predictor = Predictor(self.vas_software_path)
        # model_init_start = time.time()
        predictor.model_init()
        # log_metric("model_init_time", time.time() - model_init_start)
        return predictor

    def conduct_test_video(self, test_list, predictor):
        '''
            Model prediction interface for video
            - for each video:
                - call_vas_inference() - run model on test data stream and return results
                - add mlflow log
        '''

        for test in test_list:
            exp_name = self.get_config(['EXPERIMENT METADATA', 'EXP_NAME'])
            run_name = self.get_config(['EXPERIMENT METADATA', 'RUN_NAME'])
            set_experiment(exp_name)
            with start_run(run_name=run_name):
                result = predictor.model_infer(
                    self.get_path('TEST_DATA_PATH', test['File_name']))

                test.update(result)

                # mlflow log
                for k, v in test.items():
                    if k.startswith('meta_'):
                        log_param(k[5:], v)
                    elif not k.startswith('p_'):
                        log_param(k, v)
                    # else:
                    #         log_param(k, v)

                metrics = eval_metrics_video(
                    test_result=test,
                    metrics_to_eval=self.get_metrics_to_eval(),
                    annotation_path=self.get_path(
                        'ANNOTATION_PATH', test['File_name'][:-3] + 'xml')
                )

                for k, v in metrics.items():
                    log_metric(k, v)

        return test_list

    def conduct_test_image(self, test_list, predictor):
        '''
            Model prediction interface for image
            - for each image:
                - call_vas_inference() - run model on test data stream and return results
            - add mlflow log
        '''

        predicted_bb = []
        exp_name = self.get_config(['EXPERIMENT METADATA', 'EXP_NAME'])
        run_name = self.get_config(['EXPERIMENT METADATA', 'RUN_NAME'])
        set_experiment(exp_name)

        with start_run(run_name=run_name):
            for test in test_list:
                file_path = self.get_path('TEST_DATA_PATH', test['File_name'])
                if not os.path.exists(file_path):
                    print(
                        'Error: file not found, please check TEST_DATA_PATH in config')
                    raise Exception()
                result = predictor.model_infer(file_path)

                # only collect the results, meta data is discarded
                # test.update(result)
                file_name = file_path.split('/')[-1]
                file_name = file_name[:file_name.rfind('.')]

                dets = result['p_detections']
                for res in dets:
                    res += [file_name]
                predicted_bb += dets

            metrics = eval_metrics_image_bb(
                test_result=predicted_bb,
                metrics_to_eval=self.get_metrics_to_eval(),
                anno_folder_path=self.get_path(
                    'ANNOTATION_PATH')
            )

            for k, v in metrics.items():
                log_metric(k, v)

        return test_list

    '''
    ################################
        helper functions
    ################################
    '''

    def get_config(self, config_keys):
        result = self.config
        for key in config_keys:
            result = result[key]
        return result

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

    def get_path(self, data_type, file_name=None):
        file_dir = self.get_config(['TEST DATA', data_type])
        if not file_dir.startswith('/'):
            # relative path
            if file_dir.startswith('./'):
                file_dir = file_dir[2:]
            # file_dir = self.test_dir + '/' + file_dir
            file_dir = '/ext_vol/' + file_dir

        # if file name is not speficied, return dir
        if not file_name:
            return file_dir

        file_path = os.path.join(file_dir, file_name)
        return file_path

    def get_metrics_to_eval(self):
        return [m.lower() for m, take in self.get_config(['METRICS']).items() if take == '1']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str,
                        default=None, help='config path')
    args = parser.parse_args()
    th = VAS_Test_Handler(args.config_path)
    th.run()
