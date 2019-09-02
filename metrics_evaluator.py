import sklearn.metrics as skmetrics


def eval_metrics(test_result, metrics_to_eval):
    '''
        input: test results containing keys [gt_<v>, p_<v>]
        output: {name of metrics: evaluation result}
    '''
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
        if metric == 'explained_variance_score':
            eval_result[metric] = explained_variance_score(
                labels=labels, y_true=y_true, y_pred=y_pred)
        # TODO: add more metric eval support
        # by elif metric == 'some other metric' then ...

        else:
            print('Currently only support pre-defined metrics')
    return eval_result


def explained_variance_score(labels, y_true, y_pred):
    '''
        metrics evaluation method template
        input:
        labels: [people, pmd, bicycle] # for debug purpose maybe
        y_true: [100,200,300]
        y_pred: [100,200,300]
        the same index refers to the count/label for the same variable.
        output: evaluation results in float
    '''
    return skmetrics.explained_variance_score(y_true=y_true, y_pred=y_pred)
