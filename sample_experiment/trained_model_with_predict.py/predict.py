'''
    VAS automated performance evaluation framework
    @author: Ji Jiahao
    @date created: 20190823
    @last modified: 20191016
    @version: v0.2
    @description: predictor class example for testing mobius
    @note: 1. must implement model_init and model_infer
           2. detector tester has not been tested.
'''
from configparser import ConfigParser
from mobius.object_counter import ObjectCounterSession
from yolov3_detector.yolov3 import YOLOv3
import os
import cv2


class Predictor():
    def __init__(self, vas_path):
        self.vas_software_path = vas_path

    def __load_vas_config(self):
        ''' configuration for the model '''
        config_file = os.path.join(
            self.vas_software_path, 'cfg/lta_tunnel.ini')

        display = False
        debug = False

        return config_file, display, debug

    def model_init(self, test_type):
        if test_type == 'video':
            self.counter_model_init()
        elif test_type == 'image':
            # TODO: detector tester needs to be checked
            self.detector_model_init()

    def detector_model_init(self):
        '''
            Template method for model initialization
        '''
        config_file, display, debug = self.__load_vas_config()

        config = ConfigParser()
        config.read(config_file)
        self.model = YOLOv3(gpu_id=0)

    def detector_model_infer(self, test_image_path):
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
        config_file, display, debug = self.__load_vas_config()
        config = ConfigParser()
        config.read(config_file)

        model = YOLOv3(gpu_id=0)
        self.session = ObjectCounterSession(
            model, display=display, debug=debug, hyperparams=config['PARAMETERS'])

    def counter_model_infer(self, test_video, roi_path):

        self.session.process_video(
            test_video, roi=self.get_roi(roi_path))
        object_counts = self.session.reset_counts()

        p_ppl_count = sum(object_counts['person'].values())
        p_pmd_count = sum(object_counts['pmd'].values())
        p_bicycle_count = sum(object_counts['bicycle'].values())

        time_taken = self.session.speed_monitor.overall_time
        processing_fps = self.session.speed_monitor.avg_fps

        '''
            IMPT: return dictionary key format
            'p_<class/variable label>'
            other metadata returned shall have key format
            'meta_<metadata>'
        '''
        return {
            'p_Bicycle': p_bicycle_count,
            'p_PMD': p_pmd_count,
            'p_Person': p_ppl_count,
            'meta_time_taken': time_taken,
            'meta_processing_fps': processing_fps,
        }

    def get_roi(self, roi_path):
        if not roi_path or not os.path.exists(roi_path):
            return None
        pts = []
        with open(roi_path) as file:
            for line in file:
                pts.append([int(s) for s in line.split(',')])

        return pts


if __name__ == '__main__':
    # test_video_path = '/home/jijiahao/Documents/projects/VAS_test_data/counter_test/test_set/Cam1_2019-05-01_09-30-00_10-00-00_8fps.mp4'

    test_video_path = '/home/jijiahao/Desktop/mobius test set labelling/test_set/10_2018-05-29_20-00-00_20-05-00_8fps.mp4.mp4'
    roi_path = '/home/jijiahao/Desktop/mobius test set labelling/test_set_ROI/10_2018-05-29_20-00-00_20-05-00_roi.txt'
    # test_image_path = '/home/jijiahao/Documents/projects/VAS_test_data/detector_test/OFFICIAL_DATA/test/images/2018-09-17_14-45-00-427_16_160.jpg'

    predictor = Predictor('.')

    predictor.counter_model_init()
    print(predictor.counter_model_infer(test_video_path, roi_path))
