import pandas as pd
import Module as M
from datetime import datetime
import os
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str, default='./data')
    parser.add_argument('--ce',type=str, default='[best c&e]2023-03-11-10-13-34-c&e_model.pkl.csv')
    parser.add_argument('--wt',type=str, default='[best w&t]2023-03-10-19-32-30-w&t_model.pkl.csv')
    args = parser.parse_args()
    return args



def main(args:argparse.Namespace):
    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    name_ce = args.ce
    name_wt = args.wt
    data_path = args.data_path

    df_ce = pd.read_csv(os.path.join(data_path, name_ce))
    df_wt = pd.read_csv(os.path.join(data_path, name_wt))

    ce_labels = df_ce['label'].values
    wt_labels = df_wt['label'].values
    labels = []
    for ce_label, wt_label in zip(ce_labels, wt_labels):
        crash, ego = M.split_ce(ce_label)
        weather, timing = M.split_wt(ce_label)
        label = M.merge_label(crash, ego, weather, timing)
        labels.append(label)

    df_ce['label'] = labels
    df_ce.to_csv(f"{now}_{name_ce}_{name_wt}", index=False)

if __name__ == '__main__':
    args = parse_args()
    main(args)