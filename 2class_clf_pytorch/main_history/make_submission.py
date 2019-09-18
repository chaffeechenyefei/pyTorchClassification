import argparse
import pandas as pd
from utils import mean_df
from dataset import DATA_ROOT
import os

def get_predicts(list, weights):
    all = []
    for tmp,w in zip(list,weights):
        tmp_list = os.listdir(tmp)
        tmp_list =[[os.path.join(tmp,tmp_file),w] for tmp_file in tmp_list]
        all.extend(tmp_list)
    return all

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--threshold', type=float, default=0.1)
    args = parser.parse_args()
    dirs = [

        r'/home/ubuntu/pytorch_code/2class_clf_pytorch/result/se50_talking_0.8/12tta',
    ]

    weight = [
        1
    ]

    predicts = get_predicts(dirs,weight)
    dfs = []

    sum_w = 0
    use_pred = 0
    for prediction,w in predicts:
        print('pred', prediction)
        df = pd.read_hdf(prediction, index_col='id')
        sum_w += w
        use_pred += 1
        dfs.append(df)
    
    df = pd.concat(dfs)
    df_mean = mean_df(df)
    del df
    df_mean.index = pd.read_hdf(prediction, index_col='id').index
    df_mean = df_mean.reset_index()
    df_mean.columns = ['id','pred']
    df_mean.to_csv('tta_result.csv', index = False)


def get_classes(item):
    return ' '.join(cls for cls, is_present in item.items() if is_present)


if __name__ == '__main__':
    main()
