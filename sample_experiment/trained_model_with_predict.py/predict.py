'''
    VAS automated performance evaluation framework
    @author: Ji Jiahao
    @date created: 20190823
    @last modified: 20190911
    @version: v0.1.1
    @description: predictor class example for testing mobius
    @note: must implement model_init and model_infer
'''
from configparser import ConfigParser
from mobius.object_counter import ObjectCounterSession
from darknet.yolov3 import YOLOv3
import os
import cv2


class Predictor():
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
        self.model = YOLOv3(yolo_config, weights, yolo_meta, 0)

    def model_infer(self, test_image_path):
        '''
            Template method for model infrence
        '''
        test_image = cv2.imread(test_image_path)
        dets = self.model.detect(test_image, 0.25)
        print("\tNumber of detections: {}".format(len(dets)))

        det_boxes = []
        for i, d in enumerate(dets):
            d = list(d)
            detbox = list(d[2])

            # convert box format to cv2 tracker's format
            h = int(detbox[3])
            w = int(detbox[2])
            x = int(detbox[0] - detbox[2] / 2)
            y = int(detbox[1] - detbox[3] / 2)
            d[2] = (x, y, w, h)

            det_boxes.append(d)

        return {'p_detections': dets}

    def counter_model_init(self):
        config_file, weights, yolo_config, \
            yolo_meta, display, debug = self.__load_vas_config()
        config = ConfigParser()
        config.read(config_file)
        model = YOLOv3(yolo_config, weights, yolo_meta, 0)
        self.session = ObjectCounterSession(
            model, display=display, debug=debug, hyperparams=config['PARAMETERS'])

    def counter_model_infer(self, test_video):

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


if __name__ == '__main__':
    predictor = Predictor('.')
    predictor.model_init()
    predictor.model_infer('18092018_17-16-01-854_14_176.jpg')
