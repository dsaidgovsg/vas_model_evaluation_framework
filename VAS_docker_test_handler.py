'''
    VAS automated performance evaluation framework
    @author: Ji Jiahao
    @date created: 20190823
    @last modified: 20190911
    @version: v0.2.0
    @description: Handle tests on dockerized VAS system.
    @features: 1. mlflow used as test tracking platform
               2. support scipt testing and docker image testing

    @ UNDER DEVELOPMENT
'''

import argparse
import os
from configparser import ConfigParser

DEBUG = True


class VASDockerTestHandler():
    def __init__(self, test_config=None):
        if not test_config:
            if DEBUG:
                test_config = '/ext_vol/vas_model_evaluation_framework/sample_experiment/test_cfg_docker.ini'
            else:
                test_config = '/ext_vol/test_cfg_docker.ini'

        self.config = ConfigParser(allow_no_value=True)
        self.config.read(test_config)

        os.environ["CUDA_VISIBLE_DEVICES"] = self.get_config(
            ['ENVIRONMENT', 'GPU_ID'])
        os.environ["MLFLOW_TRACKING_URI"] = self.get_config(
            ['ENVIRONMENT', 'MLFLOW_TRACKING_URI'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str,
                        default=None, help='config path')
    args = parser.parse_args()
    th = VASDockerTestHandler(args.config_path)
    th.run()
