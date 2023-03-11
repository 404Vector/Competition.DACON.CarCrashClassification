
def get_each_acc(num_of_class:int, labels, preds):
    result = [0.0 for _ in range(num_of_class)]
    result_n = [0.0 for _ in range(num_of_class)]
    for label, pred in zip(labels, preds):
        result_n[label] += 1.
        if label != pred: continue
        result[label] += 1.
    for i in range(num_of_class):
        result[i] = result[i] / result_n[i]
    return result

def print_each_acc(acc_list):
    word = ' - acc '
    for idx, acc in enumerate(acc_list):
        word += "{}:{:.3f} / ".format(idx,acc)
    print(word)