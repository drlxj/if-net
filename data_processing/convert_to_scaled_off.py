import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
import trimesh
#import pymeshlab
from sharp_challenge1.data import *
import argparse
import tqdm

# INPUT_PATH = 'shapenet/data'

# def to_off(path):

#     if os.path.exists(path + '.off'):
#         return

#     input_file  = path + '.obj'
#     output_file = path + '.off'

#     #ms = pymeshlab.MeshSet()
#     #ms.load_new_mesh(input_file)
#     #ms.save_current_mesh(output_file)
#     ms = trimesh.load_mesh(input_file)
#     ms.export(output_file)

def scale(path):

    if os.path.exists(path + '_scaled.off'):
        print(f"exists {path}")
        return

    #try:
    input_file  = path + '.obj'

    mesh = trimesh.load(input_file, force='mesh')
    total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
    centers = (mesh.bounds[1] + mesh.bounds[0]) /2

    mesh.apply_translation(-centers)
    mesh.apply_scale(1/total_size)
    print(f'Just one step from {path}')
    mesh.export(path + '_scaled.off')
    #except:
    #    print('Error with {}'.format(path))
    print('Convert to scaled off finished {}'.format(path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process train or test data'
    )
    #parser.add_argument('-data', type=str, default='test')
    parser.add_argument('-data', type=str)
    args = parser.parse_args()

    if args.data == "train":
        INPUT_PATH = '../SHARP_data/track2/train_partial'
    elif args.data == "test":
        INPUT_PATH = '../SHARP_data/track2/test_partial'
    elif args.data == "test-codalab-partial":
        INPUT_PATH = '../SHARP_data/track2/test-codalab-partial'
    elif args.data == "train_gt":
        INPUT_PATH = '../SHARP_data/track2/train_gt'
    elif args.data == "test_gt":
        INPUT_PATH = '../SHARP_data/track2/test_gt'
    elif args.data == "val":
        INPUT_PATH = '../SHARP_data/track3/val_partial'

    p = Pool(20)
    if args.data == "val" or args.data == "train_gt" or args.data == "test_gt":
        for file in tqdm.tqdm(glob.glob(INPUT_PATH + '/*/*.obj'), desc = 'to_off'):
            #print(f"current file: {file}")
            fname = os.path.splitext(file)[0]
            if os.path.exists(fname+".obj"):
                pass
            else:
                current_mesh = load_mesh(file)
                save_obj(fname + '.obj', current_mesh, save_texture=True)
            #p.apply_async(scale,(fname,))
            scale(fname)
    else:
         for file in tqdm.tqdm(glob.glob(INPUT_PATH + '/*/*..npz'), desc = 'to_off'):
            #print(f"current file: {file}")
            fname = os.path.splitext(file)[0]
            if os.path.exists(fname+".obj"):
                pass
            else:
                current_mesh = load_mesh(file)
                save_obj(fname + '.obj', current_mesh, save_texture=True)
            p.apply_async(scale,(fname,))   
         
    # for file in tqdm.tqdm(glob.glob(INPUT_PATH + '/*/*.off'), desc = 'scale'):
    #     fname= os.path.splitext(file)[0]
    #     if os.path.exists(fname+"_scaled.off"):
    #         continue
    #     p.apply_async(scale,(fname,))
    #     pass

    p.close()
    p.join()
