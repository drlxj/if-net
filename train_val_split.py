import numpy as np
import os
from glob import glob
import random
from tqdm import tqdm

if __name__ == '__main__':
    ROOT1 = '../SHARP_data/track2/train_partial'
    train_files = glob(ROOT1 + '/*/*scaled.off')
    train_paths = []
    print(f"The number of train files: {len(train_files)}")
    for file in tqdm(train_files, desc = "train"):
        train_paths.append(os.path.splitext(file)[0])
    train = np.array(train_paths, dtype = np.str_)

    ROOT3 = '../SHARP_data/track2/test_partial'
    val_files = glob(ROOT3 + '/*/*scaled.off')
    val_paths = []
    print(f"The number of test files: {len(val_files)}")
    for file in tqdm(val_files, desc = "test" ):
        val_paths.append(os.path.splitext(file)[0])
    val = np.array(val_paths, dtype = np.str_)

    ROOT2 = '../SHARP_data/track3/val_partial'
    test_files = glob(ROOT2 + '/*/*scaled.off')
    test_paths = []
    print(f"The number of validation: {len(test_files)}")
    for file in tqdm(test_files, desc = "validation"):
        test_paths.append(os.path.splitext(file)[0])
    test = np.array(test_paths, dtype = np.str_)

    np.savez("/cluster/project/infk/courses/252-0579-00L/group20/SHARP_data/track2/split.npz", train = train, val = val, test = test)

