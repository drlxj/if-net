import numpy as np
import os
from glob import glob
import random

if __name__ == '__main__':
    ROOT1 = '../SHARP_data/track1/train_partial'
    folders = glob(ROOT1 + '/*/')
    folders = np.array(folders, dtype = np.str_)
    
    random.shuffle(folders)
    train_folders = folders[:int(folders.shape[0]*0.8)]
    val_folrders = folders[int(folders.shape[0]*0.8):]
    
    train = []
    for folder in train_folders:
        files = glob(folder + '/*scaled.off')
        for file in files:
            train.append(os.path.splitext(file)[0])
    train = np.array(train, dtype = np.str_)

    val = []
    for folder in val_folrders:
        files = glob(folder + '/*scaled.off')
        for file in files:
            val.append(os.path.splitext(file)[0])
    val = np.array(val, dtype = np.str_)

    ROOT2 = '../SHARP_data/track1/test_partial'
    test_files = glob(ROOT2 + '/*/*scaled.off')
    test_paths = []
    for file in test_files:
        test_paths.append(os.path.splitext(file)[0])
    test = np.array(test_paths, dtype = np.str_)
    print(test)

    np.savez("/cluster/project/infk/courses/252-0579-00L/group20/SHARP_data/track1/split.npz", train = train, val = val, test = test)

    

