
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

def split_wt(wt:int):
    ''' 0~6 : self.encoded_weather + (3*self.encoded_timing)
        0:'day-normal',
        1:'day-snowy',
        2:'day-rainy',
        3:'night-normal',
        4:'night-snowy',
        5:'night-rainy',
     '''
    return (wt % 3), (wt//3)
    
def split_ce(ce:int):
    ''' 0~3 : self.encoded_crash + self.encoded_ego_involve
        0: not crash
        1: crash
        2: ego+crash
    '''
    return int(ce>0), int(ce==2)

def merge_label(crash:int, ego_involve:int, weather:int, timing:int):
    label = 0
    label += ((ego_involve==0)*6)
    label += (2*weather)
    label += 1 + timing
    return crash * label