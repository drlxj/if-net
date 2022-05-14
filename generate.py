import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
import numpy as np
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from models.generation import Generator
from generation_iterator import gen_iterator
import time
import random

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def generate_basic(rank, exp_name, world_size, args):
    print(f"Running basic DDP on rank {rank}.")
    setup(rank, world_size)
    
    if args.model ==  'ShapeNet32Vox':
        net = model.ShapeNet32Vox()

    if args.model ==  'ShapeNet128Vox':
        net = model.ShapeNet128Vox()

    if args.model == 'ShapeNetPoints':
        net = model.ShapeNetPoints()

    if args.model == 'SVR':
        net = model.SVR()


    dataset = voxelized_data.VoxelizedDataset(args.mode, voxelized_pointcloud= args.pointcloud , pointcloud_samples= args.pc_samples, res=args.res, sample_distribution=args.sample_distribution,
                                            sample_sigmas=args.sample_sigmas ,num_sample_points=100, batch_size=1, num_workers=0)

    # Dataset index shuffle
    gen_length = len(dataset)
    gen_partial_length = int(gen_length/world_size)
    if not rank:
        #dist.send(tensor=torch.Tensor(val_loss), dst=1)
        gen_index = np.arange(0,gen_length, dtype = int)
        np.random.shuffle(gen_index)
        for i in range(1, world_size):
            partial_index = torch.from_numpy(gen_index[gen_partial_length*i: gen_partial_length*(i+1)])
            dist.send(tensor=partial_index.to(dtype = torch.int), dst=i)
        gen_index = gen_index[: gen_partial_length]
    else:
        index_torch = torch.zeros(gen_partial_length,dtype=torch.int)
        dist.recv(tensor=index_torch, src=0)
        gen_index = index_torch.detach().cpu().numpy()
    dist.barrier()
    
    dataset.random_split(gen_index)


    gen = Generator(net,0.5, exp_name, checkpoint=args.checkpoint ,resolution=args.retrieval_res, batch_points=args.batch_points)

    out_path = 'experiments/{}/evaluation_{}_@{}/'.format(exp_name,args.checkpoint, args.retrieval_res)

    gen_iterator(out_path, dataset, gen)

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run generation'
    )


    parser.add_argument('-pointcloud', dest='pointcloud', action='store_true')
    parser.add_argument('-voxels', dest='pointcloud', action='store_false')
    parser.set_defaults(pointcloud=False)
    parser.add_argument('-pc_samples' , default=3000, type=int)
    parser.add_argument('-dist','--sample_distribution', default=[0.5,0.5], nargs='+', type=float)
    parser.add_argument('-std_dev','--sample_sigmas',default=[], nargs='+', type=float)
    parser.add_argument('-res' , default=32, type=int)
    parser.add_argument('-decoder_hidden_dim' , default=256, type=int)
    parser.add_argument('-mode' , default='test', type=str)
    parser.add_argument('-retrieval_res' , default=256, type=int)
    parser.add_argument('-checkpoint', type=int)
    parser.add_argument('-batch_points', default=1000000, type=int)
    parser.add_argument('-m','--model' , default='LocNet', type=str)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_known_args()[0]

    exp_name = 'i{}_dist-{}sigmas-{}v{}_m{}'.format(  'PC' + str(args.pc_samples) if args.pointcloud else 'Voxels',
                                        ''.join(str(e)+'_' for e in args.sample_distribution),
                                        ''.join(str(e) +'_'for e in args.sample_sigmas),
                                                                    args.res,args.model)
    # exp_name = 'Trained_Models/{}'.format(args.model)

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus

    random.seed(time.time())

    #net = None
    #args = None
    
    mp.spawn(generate_basic,
             args=(exp_name, world_size, args),
             nprocs=world_size,
             join=True)
    print("main finished")

    
