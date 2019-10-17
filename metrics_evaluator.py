import sklearn.metrics as skmetrics
import xml.etree.ElementTree as ETree
from glob import glob
import numpy as np


def eval_metrics_video(test_result, metrics_to_eval, annotation_path='none', overall_metric=False):
    '''
        input: test results containing keys [gt_<v>, p_<v>] or,
               test results containing [p_<v>] and annotation path
        output: {name of metrics: evaluation result}
    '''

    if overall_metric:
        print('overall metrics to evaluate:')
    else:
        print('metrics to evaluate:')
    print(metrics_to_eval)

    # sort out the test result, prepare for evaluation
    labels = []
    y_true = []
    y_pred = []

    for key, value in test_result.items():
        if key.startswith('gt_'):
            labels.append(key[3:])
            y_true.append(value)
            y_pred.append(test_result['p_' + key[3:]])

    # evaluate each of the metrics and save in a dictionary
    eval_result = {}
    for metric in metrics_to_eval:
        if overall_metric:
            # overall metrics
            if 'explained_variance_score' in metric:
                # score for each label class
                for label_idx in range(len(labels)):
                    eval_result[metric + '_' + labels[label_idx]] = explained_variance_score(
                        labels=labels, y_true=y_true[label_idx], y_pred=y_pred[label_idx])
                eval_result['overall_' + metric] = explained_variance_score(
                    labels=labels, y_true=y_true, y_pred=y_pred)
            elif 'f1_score' in metric:
                eval_result.update(
                    f1_score(labels=labels, y_true=y_true, y_pred=y_pred))

            else:
                raise ValueError('Unsuppported metrics: {}'.format(metric))
        else:
            # individual metrices
            if metric == 'explained_variance_score':
                eval_result[metric] = explained_variance_score(
                    labels=labels, y_true=y_true, y_pred=y_pred)
    return eval_result


def f1_score(labels, y_true, y_pred):
    def count(gt, p, miss_or_false='miss'):
        if miss_or_false == 'miss':
            miss_count = gt - p
            miss_count[miss_count < 0] = 0
            return miss_count
        elif miss_or_false == 'false':
            false_count = p - gt
            false_count[false_count < 0] = 0
            return false_count

    def precision(gt, false_count):
        return np.sum(gt) / (np.sum(gt) + np.sum(false_count))

    def recall(gt, miss_count):
        return np.sum(gt) / (np.sum(gt) + np.sum(miss_count))

    def f1(prec, rec):
        return 2 * prec * rec / (prec + rec)

    res = {}
    gt_sum = 0
    false_count_sum = 0
    miss_count_sum = 0

    for label_idx in range(len(labels)):
        class_gt_array = np.array(y_true[label_idx])
        class_p_array = np.array(y_pred[label_idx])

        false_count = count(class_gt_array, class_p_array, 'false')
        miss_count = count(class_gt_array, class_p_array, 'miss')

        prec = precision(class_gt_array, false_count)
        rec = recall(class_gt_array, miss_count)
        res['precision_' + labels[label_idx]] = prec
        res['recall_' + labels[label_idx]] = rec
        res['f1_' + labels[label_idx]] = f1(prec, rec)

        gt_sum += np.sum(class_gt_array)
        false_count_sum += np.sum(false_count)
        miss_count_sum += np.sum(miss_count)

    res['overall_precision'] = precision(gt_sum, false_count_sum)
    res['overall_recall'] = recall(gt_sum, miss_count_sum)
    res['overall_f1'] = f1(res['overall_precision'], res['overall_recall'])

    return res


def explained_variance_score(labels, y_true, y_pred, multioutput='variance_weighted'):
    '''
        metrics evaluation method template
        input:
        labels: [people, pmd, bicycle]
        y_true: [100,200,300]
        y_pred: [100,200,300]
        the same index refers to the count/label for the same variable.
        output: evaluation results in float
    '''
    return skmetrics.explained_variance_score(y_true=y_true, y_pred=y_pred, multioutput=multioutput)


def eval_metrics_image_bb(test_result, metrics_to_eval, anno_folder_path):
    print('metrics to evaluate:')
    print(metrics_to_eval)
    # get annotations
    annotation = _read_all_annotation(anno_folder_path)

    # evaluate each of the metrics and save in a dictionary
    eval_result = {}
    for metric in metrics_to_eval:
        if metric == 'mean_average_precision':
            eval_result[metric] = mean_average_precision(
                # TODO: put definition of labels to test_cfg.ini
                labels=["pmd", "bicycle", "person"],
                p_detections=test_result,
                annotations=annotation)

    return eval_result


def _read_all_annotation(anno_folder_path):
    all_xml_file = [f for f in glob(
        anno_folder_path + '/*') if f.endswith('.xml')]

    annotations = {}
    for file in all_xml_file:
        annotations[file.split('/')[-1][:-4]] = _read_one_annotation(file)

    return annotations


def _read_one_annotation(annotation_path, classes=["pmd", "bicycle", "person"]):
    with open(annotation_path) as in_file:
        try:
            tree = ETree.parse(in_file)
        except Exception as e:
            print(e)

    bboxes = []
    root = tree.getroot()
    # size = root.find('size')
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)

    for obj in root.iter('object'):
        if obj.find('difficult').text == 1:
            continue
        cls = obj.find('name').text

        xmlbox = obj.find('bndbox')
        b = [
            float(xmlbox.find('xmin').text), float(
                xmlbox.find('xmax').text),
            float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text)
        ]
        # class label, bbox, used_flag
        bboxes.append([cls, b, False])

    return bboxes


def mean_average_precision(labels, p_detections, annotations, min_overlap=0.5):
    '''
        Pascal VOC object detection evaluation metrics

        script adapted from https://github.com/Cartucho/mAP,
        a python implementation of the official Matlab code for PASCAL VOC 2012 competition

        input python box coord: (left, right, top, bottom)

            0,0 ------> x (width)
             |
             |  (Left,Top)
             |      *_________
             |      |         |
                    |         |
             y      |_________|
          (height)            *
                        (Right,Bottom)

        input format:
            labels: list of class labels
            p_detections: list of detections
                        [[label, confidence, bbox, file_name], ...]
            annotations: dictionary
                        {file_name: list of GTs - [[label, bbox, used_flag], ...]}
            min_overlap: minimun IOU to be counted as tp
        Calculate the AP given the recall and precision array
            1st) We compute a version of the measured precision/recall curve with
                 precision monotonically decreasing
            2nd) We compute the AP as the area under this curve by numerical integration.
    '''

    def voc_ap(rec, prec):
        """
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
                mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        rec.insert(0, 0.0)  # insert 0.0 at begining of list
        rec.append(1.0)  # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0)  # insert 0.0 at begining of list
        prec.append(0.0)  # insert 0.0 at end of list
        mpre = prec[:]
        """
         This part makes the precision monotonically decreasing
            (goes from the end to the beginning)
            matlab: for i=numel(mpre)-1:-1:1
                        mpre(i)=max(mpre(i),mpre(i+1));
        """
        # matlab indexes start in 1 but python in 0, so I have to do:
        #     range(start=(len(mpre) - 2), end=0, step=-1)
        # also the python function range excludes the end, resulting in:
        #     range(start=(len(mpre) - 2), end=-1, step=-1)
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        """
         This part creates a list of indexes where the recall changes
            matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)  # if it was matlab would be i + 1
        """
         The Average Precision (AP) is the area under the curve
            (numerical integration)
            matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
        return ap, mrec, mpre

    def _convert(box):
        '''
            change coord, from center (x,y) and (w,h) to (l,r,t,b)
        '''
        lf = box[0] - box[2] / 2
        rt = box[0] + box[2] / 2
        top = box[1] - box[3] / 2
        btm = box[1] + box[3] / 2

        return (lf, rt, top, btm) if isinstance(box, tuple) else [lf, rt, top, btm]

    '''
        ground truth processing
    '''
    gt_counter_per_class = {}

    for gts in annotations.values():
        for gt in gts:
            if gt[0] in gt_counter_per_class.keys():
                gt_counter_per_class[gt[0]] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[gt[0]] = 1

    print('\nground truth stats')
    print(gt_counter_per_class)
    for k, v in annotations.items():
        print('\n{}: {}'.format(k, v))
        break

    '''
        detection result preprocessing
    '''

    # convert label and bbox format
    for i in range(len(p_detections)):
        p_detections[i][0] = labels[p_detections[i][0]]
        p_detections[i][2] = _convert(p_detections[i][2])

    det_boxes = {}
    for label in labels:
        det_per_class = [det for det in p_detections if det[0] == label]
        det_per_class.sort(key=lambda x: x[1], reverse=True)
        det_boxes[label] = det_boxes.get(label, []) + det_per_class

    print('\nprediction stats')
    for label in labels:
        print(det_boxes[label][0])
        print('{}, count: {} \n'.format(label, len(det_boxes[label])))

        # DEBUG
        print(annotations[det_boxes[label][0][3]])
        print('')

    """
        calculate the AP for each class
    """
    sum_AP = 0.0
    count_true_positives = {}
    for label in labels:
        count_true_positives[label] = 0

        """
         Assign detection-results to ground-truth objects
        """
        nd = len(det_boxes[label])
        tp = [0] * nd  # creates an array of zeros of size nd
        fp = [0] * nd

        for idx, detection in enumerate(det_boxes[label]):
            file_id = detection[3]
            ovmax = -1
            gt_match = -1
            # convert (l, r, t, b) to (l, t, r, b)
            bb = list()
            bb.append(detection[2][0])
            bb.append(detection[2][2])
            bb.append(detection[2][1])
            bb.append(detection[2][3])

            for obj in annotations[file_id]:
                if obj[0] == label:
                    # convert (l, r, t, b) to (l, t, r, b)
                    bbgt = list()
                    bbgt.append(obj[1][0])
                    bbgt.append(obj[1][2])
                    bbgt.append(obj[1][1])
                    bbgt.append(obj[1][3])

                    bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]),
                          min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + \
                            (bbgt[2] - bbgt[0] + 1) * \
                            (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            if ovmax >= min_overlap:
                if not gt_match[2]:
                    # true positive
                    tp[idx] = 1
                    gt_match[2] = True
                    count_true_positives[label] += 1
                else:
                    # false positive (multiple detection)
                    fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1

        # print(tp)
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        # print(tp)
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[label]
        # print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        # print(prec)

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap

    '''
        finally mAP
    '''
    mAP = sum_AP / len(labels)

    # DEBUG
    # print('\nmAP')
    # print(mAP)
    # import sys
    # sys.exit()

    return mAP
