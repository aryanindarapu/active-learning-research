import numpy as np

# metrics assume predictions and ground-truths are Boolean or binary-valued

def precision(pred, gt):
    true_positives = np.sum(pred*gt)
    total_predictions = np.sum(pred)
    return true_positives/max(total_predictions, 1)

def recall(pred, gt):
    true_positives = np.sum(pred*gt)
    total_positives = np.sum(gt)
    if total_positives:
        return true_positives/total_positives
    else:
        return 1
