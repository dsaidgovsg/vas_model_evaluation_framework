from configparser import ConfigParser
from mobius.object_counter import ObjectCounterSession
from darknet.yolov3 import YOLOv3
import os


class Predictor():
    '''
        predictor class example for testing mobius
    '''

    def __init__(self, vas_path):
        self.vas_software_path = vas_path

    def __load_vas_config(self):
        ''' configuration for the model '''
        config_file = os.path.join(
            self.vas_software_path, 'cfg/lta_roads_cfg.ini')
        weights = os.path.join(self.vas_software_path,
                               'darknet/yolov3-cm_12000.weights')
        yolo_config = os.path.join(
            self.vas_software_path, 'darknet/yolov3-cm_test.cfg')
        yolo_meta = os.path.join(self.vas_software_path, 'darknet/cm.data')
        display = False
        debug = True

        return config_file, weights, yolo_config, \
            yolo_meta, display, debug

    def model_init(self):
        '''
            Template method for model initialization
        '''
        config_file, weights, yolo_config, \
            yolo_meta, display, debug = self.__load_vas_config()
        config = ConfigParser()
        config.read(config_file)
        model = YOLOv3(yolo_config, weights, yolo_meta, 0)
        self.session = ObjectCounterSession(
            model, display=display, debug=debug, hyperparams=config['PARAMETERS'])

    def model_infer(self, test_video):
        '''
            Template method for model infrence
        '''
        time_taken, processing_fps = self.session.process_video(test_video)
        object_counts = self.session.reset_counts()

        p_ppl_count = sum(object_counts['person'].values())
        p_pmd_count = sum(object_counts['pmd'].values())
        p_bicycle_count = sum(object_counts['bicycle'].values())

        '''
            IMPT: return dictionary key format
            'p_<class/variable label>'
            other metadata returned shall have key format
            'meta_<metadata>'
        '''
        return {
            'p_Bicycle': p_bicycle_count,
            'p_PMD': p_pmd_count,
            'p_People': p_ppl_count,
            'meta_time_taken': time_taken,
            'meta_processing_fps': processing_fps,
        }
