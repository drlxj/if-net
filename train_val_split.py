import numpy as np
import os
from glob import glob
import random

if __name__ == '__main__':
    ROOT1 = '../SHARP_data/track2/train_partial'
    train_files = glob(ROOT1 + '/*/*scaled.off')
    train_paths = []
    for file in train_files:
        train_files.append(os.path.splitext(file)[0])
    train = np.array(train_paths, dtype = np.str_)

    ROOT3 = '../SHARP_data/track2/test_partial'
    train_files = glob(ROOT1 + '/*/*scaled.off')
    train_paths = []
    for file in train_files:
        train_files.append(os.path.splitext(file)[0])
    val = np.array(train_paths, dtype = np.str_)

    ROOT2 = '../SHARP_data/track3/val_partial'
    test_files = glob(ROOT2 + '/*/*scaled.off')
    test_paths = []
    for file in test_files:
        test_paths.append(os.path.splitext(file)[0])
    test = np.array(test_paths, dtype = np.str_)

    np.savez("/cluster/project/infk/courses/252-0579-00L/group20/SHARP_data/track2/split.npz", train = train, val = val, test = test)
    

