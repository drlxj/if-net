import trimesh
import numpy as np
import implicit_waterproofing as iw
import glob
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import os
import traceback


def boundary_sampling(tmp_path):
    path, off_path, args, sample_num = tmp_path
    try:

        off_path = off_path
        fname = os.path.splitext(off_path)[0]
        out_file = fname +'_boundary_{}_samples.npz'.format(args.sigma)

        # if os.path.exists(out_file):
        #     print('Boundary file exists. Done.')
        #     return

        mesh = trimesh.load(off_path) # trimesh.Trimesh
        points = mesh.sample(sample_num) # Return random samples distributed across the surface of the mesh
              
        boundary_points = points + args.sigma * np.random.randn(sample_num, 3)
        grid_coords = boundary_points.copy()
        grid_coords[:, 0], grid_coords[:, 2] = boundary_points[:, 2], boundary_points[:, 0]

        grid_coords = 2 * grid_coords
        
        if args.data == "val":
            mesh_gt = trimesh.load(off_path)
        else:
            root = os.path.split(off_path)[0]
            last = root.split(os.sep)[-1]
            second_last = root.split(os.sep)[-2]
            second_last = second_last[:-8] + "_gt"
            mesh_gt = trimesh.load(os.path.join(root,"..","..",second_last,last,last+"_scaled.off"))
        
        occupancies = iw.implicit_waterproofing(mesh_gt, boundary_points)[0]

        np.savez(out_file, points=boundary_points, occupancies = occupancies, grid_coords= grid_coords)
        print('Finished {}'.format(path))
    except:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('-sigma', type=float)
    parser.add_argument('-data', type=str)

    args = parser.parse_args()

    if args.data == "train":
        ROOT = '../SHARP_data/track2/train_partial'
    elif args.data == "test":
        ROOT = '../SHARP_data/track2/test_partial'
    elif args.data == "test-codalab-partial":
        ROOT = '../SHARP_data/track2/test-codalab-partial'
    elif args.data == "val":
        ROOT = '../SHARP_data/track3/val_partial'


    sample_num = 100000

    off_paths = glob.glob(ROOT + '/*/*scaled.off')
    new_paths = []
    for off_path in off_paths:
        path = os.path.split(off_path)[0]
        new_paths.append((path, off_path, args, sample_num))


    p = Pool(mp.cpu_count())
    p.map(boundary_sampling, new_paths)
