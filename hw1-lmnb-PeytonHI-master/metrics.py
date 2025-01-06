def precision(truth: list, predictions: list) -> float:
    # raise NotImplementedError()

    true_positive = 0
    false_positive = 0
    positive_label = "SH"

    for t, p in zip(truth, predictions):
        if t == positive_label and p == positive_label: # gold label equals predicted label "SH"
            true_positive += 1
        else:
            if t != positive_label and p == positive_label: # gold label is not "SH" and predicted label is "SH"
                false_positive += 1

    precision = true_positive/ (false_positive + true_positive) # tp/ (fp+ tp)

    return precision


def recall(truth: list, predictions: list) -> float:
    # raise NotImplementedError()

    true_positive = 0
    false_negative = 0
    positive_label = "SH"

    for t, p in zip(truth, predictions):
        if t == positive_label and p == positive_label: # gold label equals predicted label "SH"
            true_positive += 1
          
        else:
            if t == positive_label and p != positive_label: # gold label is "SH" and predicted label is not "SH"
                false_negative += 1

    recall = true_positive / (true_positive + false_negative) # tp/(tp + fn)

    return recall


def f1score(truth: list, predictions: list) -> float:
    # raise NotImplementedError()

    total_precision = precision(truth, predictions)
    total_recall = recall(truth, predictions)

    f1 = (2*total_precision*total_recall) / (total_precision + total_recall) # 2PR/P+R

    return f1
