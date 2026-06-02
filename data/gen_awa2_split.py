import os
import glob
import pandas as pd
import numpy as np

def _train_test_split(x, y, test_size=0.5, random_state=None):
    n = len(x)
    if len(y) != n:
        raise ValueError("x and y must have the same length")
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")
    rng = np.random.RandomState(random_state) if random_state is not None else np.random
    idx = np.arange(n)
    rng.shuffle(idx)
    t = int(np.ceil(n * float(test_size)))
    if t <= 0:
        t = 1
    if t >= n:
        t = n - 1
    test_idx = idx[:t]
    train_idx = idx[t:]
    x_train = [x[i] for i in train_idx]
    x_test = [x[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    return x_train, x_test, y_train, y_test


def split_data(data_root,val_split=0.2, test_split=0.5):
    class_to_index = dict()
    # Build dictionary of indices to classes
    with open(f"{data_root}/classes.txt") as f:
      index = 1
      for line in f:
        class_name = line.split('\t')[1].strip()
        class_to_index[class_name] = index
        index += 1
    class_to_index = class_to_index
    
    img_names = []
    img_index = []
    for c in class_to_index.keys():
        class_name = c
        FOLDER_DIR = os.path.join(f'{data_root}/JPEGImages', class_name)
        file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
        files = glob.glob(file_descriptor)

        class_index = class_to_index[class_name]
        for file_name in files:
            img_names.append(file_name)
            img_index.append(class_index)
    img_names = img_names
    img_index = img_index
    
    # Split data into train and test
    train_img_names, test_img_names, train_img_index, test_img_index = _train_test_split(img_names, img_index, test_size=test_split, random_state=42)
    train_img_names, val_img_names, train_img_index, val_img_index = _train_test_split(train_img_names, train_img_index, test_size=val_split, random_state=42)
    
    train_df = pd.DataFrame({'img_name': train_img_names, 'img_index': train_img_index})
    val_df = pd.DataFrame({'img_name': val_img_names, 'img_index': val_img_index})
    test_df = pd.DataFrame({'img_name': test_img_names, 'img_index': test_img_index})
    pd.DataFrame.to_csv(train_df, f'{data_root}/train.csv')
    pd.DataFrame.to_csv(val_df, f'{data_root}/val.csv')
    pd.DataFrame.to_csv(test_df, f'{data_root}/test.csv')
    return train_df, val_df, test_df

if __name__ == '__main__':
    split_data("D:/BaiduNetdiskDownload/OTCBM-main/AwA2-data/Animals_with_Attributes2")
