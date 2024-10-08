'''
    VAS automated performance evaluation framework
    @author: Ji Jiahao
    @date created: 20190823
    @last modified: 20190911
    @version: v0.2.0
    @description: Handler that handles genreal tests on VAS system.
    @features: 1. mlflow used as test tracking platform
               2. support scipt testing and docker image testing

    @bug: 1. the way of calculating the explained variance score is wrong.

    @enhancement: 1. create base class and inherit vas test handler for 1. video, 2. image
'''

import os
import sys
from configparser import ConfigParser
import pandas as pd
from copy import deepcopy
from datetime import datetime
import time
from metrics_evaluator import eval_metrics_video, eval_metrics_image_bb
from mlflow import log_metric, log_param, start_run, set_experiment
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class VASTestHandler():
    def __init__(self, debug=True, test_config=None):
        self.debug_flag = debug
        if not test_config:
            if self.debug_flag == 1:
                # for debugging on dgx
                test_config = '/ext_vol/test_api_test/test_cfg.ini'
            else:
                # for deployment in docker image
                test_config = '/ext_vol/test_cfg.ini'

        print(self.debug_flag)
        print(test_config)
        if not os.path.exists(test_config):
            raise ValueError(
                'config file path not valid, config path: {}'.format(test_config))
        self.config = ConfigParser(allow_no_value=True)
        self.config.read(test_config)

        os.environ["CUDA_VISIBLE_DEVICES"] = self.get_config(
            ['ENVIRONMENT', 'GPU_ID'])
        os.environ["MLFLOW_TRACKING_URI"] = self.get_config(
            ['ENVIRONMENT', 'MLFLOW_TRACKING_URI'])

        # self.test_dir = os.getcwd()
        self.vas_software_path = self.get_path(
            'PATH', config_header='VAS SOFTWARE')
        # sys.path.append(self.test_dir)
        sys.path.append(self.vas_software_path)

    def run(self):
        test_list = self.get_test_list(
            self.get_path('DATA_SUMMARY_EXCEL_PATH'))

        ''' IMPT: change working dir'''
        os.chdir(self.vas_software_path)

        test_type = self.get_config(['TEST DATA', 'TEST_DATA_TYPE']).lower()
        predictor = self.init_test(test_type)

        if test_type == 'video':
            _ = self.conduct_test_video(test_list, predictor)
        elif test_type == 'image':
            _ = self.conduct_test_image(test_list, predictor)

        # self.__test_mlflow()

    def init_test(self, test_type):
        '''
            - init_vas_model() - initialize model once and return the active session
        '''

        self.overall_res = {'sheet_name': 'raw_results'}
        from predict import Predictor
        predictor = Predictor(self.vas_software_path)
        # model_init_start = time.time()
        predictor.model_init(test_type)
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
                try:
                    # print(self.get_roi_path(test['File_name']))
                    result = predictor.counter_model_infer(
                        self.get_path('TEST_DATA_PATH', test['File_name']),
                        roi_path=self.get_roi_path(test['File_name']))
                    result['meta_test_status'] = 'successful'
                except Exception as e:
                    print(e)
                    print('Test Failed. could not run inference on the test data.')
                    result = {'meta_test_status': 'failed'}

                test.update(result)

                # mlflow log
                for k, v in test.items():
                    if k.startswith('meta_'):
                        log_param(k[5:], v)
                        self.overall_res[k[5:]] = self.overall_res.get(
                            k[5:], []) + [v]
                    # elif not k.startswith('p_'):
                    #     log_param(k, v)
                    else:
                        log_param(k, v)
                        self.overall_res[k] = self.overall_res.get(k, []) + [v]

                if result['meta_test_status'] == 'successful':
                    metrics = eval_metrics_video(
                        test_result=test,
                        metrics_to_eval=self.get_metrics_to_eval(
                            overall_metric=False),
                        overall_metric=False
                    )

                    # print metrics value
                    print(metrics)

                    for k, v in metrics.items():
                        log_metric(k, v)

        overall_metric = eval_metrics_video(
            test_result=self.overall_res,
            metrics_to_eval=self.get_metrics_to_eval(overall_metric=True),
            overall_metric=True)

        overall_metric = {k: [v] for k, v in overall_metric.items()}
        overall_metric['sheet_name'] = 'metrics'

        # save as xlsx
        self.export_as_excel(self.overall_res, overall_metric)

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
                result = predictor.detector_model_infer(file_path)

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

    def get_path(self, data_type, file_name=None, config_header='TEST DATA'):
        file_dir = self.get_config([config_header, data_type])
        if not file_dir.startswith('/'):
            # relative path
            if file_dir.startswith('./'):
                file_dir = file_dir[2:]
            # file_dir = self.test_dir + '/' + file_dir
            if self.debug_flag == 1:
                file_dir = '/ext_vol/test_api_test/' + file_dir
            else:
                file_dir = '/ext_vol/' + file_dir

        if not os.path.exists(file_dir):
            raise ValueError('dir does not exist: {}'.format(file_dir))

        # if file name is not specified, return dir
        if not file_name:
            return file_dir

        file_path = os.path.join(file_dir, file_name)
        # if not os.path.exists(file_path):
        #     raise ValueError('file does not exist: {}'.format(file_path))
        print(file_path)
        return file_path

    def get_roi_path(self, video_name):
        if self.get_config(['TEST DATA', 'TEST_DATA_ROI_PATH']).lower() == 'none':
            return 'none'

        roi_file_name = video_name[:video_name.rfind('_')] + '_roi.txt'
        roi_file_path = self.get_path('TEST_DATA_ROI_PATH', roi_file_name)

        if not os.path.exists(roi_file_path):
            print('cannot find roi file, trying again')
            roi_file_name = video_name[:video_name.rfind('.')] + '_roi.txt'
            roi_file_path = self.get_path('TEST_DATA_ROI_PATH', roi_file_name)

        return roi_file_path

    def get_metrics_to_eval(self, overall_metric=False):
        all_metrics = [m.lower() for m, take in self.get_config(
            ['METRICS']).items() if take == '1']
        metrics_to_eval = [m for m in all_metrics if m.startswith('overall')] if overall_metric else \
            [m[8:] for m in all_metrics if not m.startswith('overall')]
        return metrics_to_eval

    def export_as_excel(self, *argv):
        now = str(datetime.now())
        writer = pd.ExcelWriter(os.path.join(
            self.get_path('OVERALL_EXCEL_STAT_SAVE_PATH',
                          config_header='ENVIRONMENT'),
            'test_results_{}.xlsx'.format(now)),
            engine='xlsxwriter')

        for arg in argv:
            sheet_name = arg.pop('sheet_name', None)
            pd.DataFrame(arg).to_excel(writer, sheet_name=sheet_name)

        writer.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', type=int,
                        default=1, help='debug flag, Set to False when deploy.')
    parser.add_argument('-c', '--config_path', type=str,
                        default=None, help='config path')
    args = parser.parse_args()
    th = VASTestHandler(debug=args.debug, test_config=args.config_path)
    th.run()
