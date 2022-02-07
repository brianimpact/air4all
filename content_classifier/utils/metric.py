import math
import numpy as np


def f1_scores(labels, logits, predictions, label_ids, label_sequences, mode=[0, 0, 0]):
    modified_predictions = np.copy(predictions)
    for i in range(labels.shape[0]):
        if mode[0] == 1:
            modified_predictions[i, :] = remove_isolated(modified_predictions[i, :], logits[i, :], label_ids, label_sequences)
        if mode[0] == 2:
            modified_predictions[i, :] = connect_isolated(modified_predictions[i, :], logits[i, :], label_ids, label_sequences)
        if mode[1] == 1:
            modified_predictions[i, :] = remove_dangling(modified_predictions[i, :], logits[i, :], label_ids, label_sequences)
        if mode[2] == 1:
            modified_predictions[i, :] = select_argmax_su(modified_predictions[i, :], logits[i, :], label_ids, label_sequences)
        if mode[2] == 2:
            modified_predictions[i, :] = select_argmax_path(modified_predictions[i, :], logits[i, :], label_ids, label_sequences)
            
    true_positive = (labels * (labels == modified_predictions)).sum(0)
    true_negative = ((1. - labels) * (labels == modified_predictions)).sum(0)
    false_positive = ((1. - labels) * (labels != modified_predictions)).sum(0)
    false_negative = (labels * (labels != modified_predictions)).sum(0)
    
    micro_precision = (true_positive.sum() / (true_positive.sum() + false_positive.sum())).item()
    micro_recall = (true_positive.sum() / (true_positive.sum() + false_negative.sum())).item()
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

    precisions = (true_positive / (true_positive + false_positive + 1e-8))[true_positive + false_positive + false_negative > 0]
    recalls = (true_positive / (true_positive + false_negative + 1e-8))[true_positive + false_positive + false_negative > 0]

    macro_precision = precisions.mean().item()
    macro_recall = recalls.mean().item()
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)

    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    average_f1 = f1s.mean().item()

    return {'micro-f1': micro_f1, 'macro-f1': macro_f1, 'average-f1':average_f1}

# post-processing for mandatory leaf assumption
def remove_isolated(prediction, logits, label_ids, paths):
    for path in paths:
        if prediction[path[0]]:
            for topic in path[1:]:
                if not prediction[topic]:
                    prediction[path[0]] = False
                    break
    return prediction

def connect_isolated(prediction, logits, label_ids, paths):
    for path in paths:
        if prediction[path[0]]:
            for topic in path[1:]:
                prediction[topic] = True
    return prediction

def remove_dangling(prediction, logits, label_ids, paths):
    dangling = [True for _ in label_ids]
    for path in paths:
        if prediction[path[0]]:
            for topic in path:
                dangling[topic] = False
    for label in range(len(label_ids)):
        if dangling[label]:
            prediction[label] = False
    return prediction

def select_argmax_su(prediction, logits, label_ids, paths):
    flag = True
    best_p = None
    best_path = []
    for path in paths:
        if prediction[path[0]]:
            flag = False
            break
        else:
            p = logits[path[0]]
            if best_p is None or p > best_p:
                best_p = p
                best_path = path
    if flag:
        for topic in best_path:
            prediction[topic] = True
    return prediction

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def select_argmax_path(prediction, logits, label_ids, paths):
    flag = True
    best_p = None
    best_path = []
    for path in paths:
        p = 1
        if prediction[path[0]]:
            flag = False
            break
        else:
            for topic in path:
                p = p * sigmoid(logits[topic])
            p = p ** (1. / len(path))
        if best_p is None or p > best_p:
            best_p = p
            best_path = path
    if flag:
        for topic in best_path:
            prediction[topic] = True
    return prediction
