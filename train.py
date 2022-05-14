import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
from models import training
import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import random
import time
import torch.nn as nn

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_basic_backup(rank, net, exp_name, world_size, args, train_index_total, val_index_total):
    setup(rank, world_size)
    cleanup()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def train_basic(rank, exp_name, world_size, args):
    print(f"Running basic DDP on rank {rank}.")
    setup(rank, world_size)
    
    if args.model ==  'ShapeNet32Vox':
        net = model.ShapeNet32Vox(rank = rank)
        #net = ToyModel().to(rank)

    if args.model ==  'ShapeNet128Vox':
        net = model.ShapeNet128Vox(rank = rank)

    if args.model == 'ShapeNetPoints':
        net = model.ShapeNetPoints(rank = rank)

    if args.model == 'SVR':
        net = model.SVR(rank = rank)
    net = net.to(rank)
    ddp_model = DDP(net, device_ids = [rank])
    if torch.cuda.get_device_name(rank) == " NVIDIA GeForce GTX 1080":
        args.batch_size = int(args.batch_size/11.0*8)
    train_dataset = voxelized_data.VoxelizedDataset('train', voxelized_pointcloud= args.pointcloud, pointcloud_samples= args.pc_samples, res=args.res, sample_distribution=args.sample_distribution,
                                           sample_sigmas=args.sample_sigmas ,num_sample_points=50000, batch_size=args.batch_size, num_workers=0, world_size = world_size, rank = rank)
    val_dataset = voxelized_data.VoxelizedDataset('val', voxelized_pointcloud= args.pointcloud , pointcloud_samples= args.pc_samples, res=args.res, sample_distribution=args.sample_distribution,
                                           sample_sigmas=args.sample_sigmas ,num_sample_points=50000, batch_size=args.batch_size, num_workers=0, world_size = world_size, rank = rank)   

    # Train index shuffle
    train_length = len(train_dataset)
    train_partial_length = int(train_length/world_size)
    if not rank:
        #dist.send(tensor=torch.Tensor(val_loss), dst=1)
        train_index = np.arange(0,train_length, dtype = int)
        np.random.shuffle(train_index)
        for i in range(1, world_size):
            partial_index = torch.from_numpy(train_index[train_partial_length*i: train_partial_length*(i+1)])
            dist.send(tensor=partial_index.to(dtype = torch.int), dst=i)
        train_index = train_index[: train_partial_length]
    else:
        index_torch = torch.zeros(train_partial_length,dtype=torch.int)
        dist.recv(tensor=index_torch, src=0)
        train_index = index_torch.detach().cpu().numpy()
    dist.barrier()

    # Validation index shuffle
    val_length = len(val_dataset)
    val_partial_length = int(val_length/world_size)
    if not rank:
        val_index = np.arange(0,val_length, dtype = int)
        np.random.shuffle(val_index)
        for i in range(1, world_size):
            partial_index = torch.from_numpy(val_index[val_partial_length*i: val_partial_length*(i+1)])
            dist.send(tensor=partial_index.to(dtype = torch.int), dst=i)
        val_index = val_index[: val_partial_length]
    else:
        index_torch = torch.zeros(val_partial_length,dtype=torch.int)
        dist.recv(tensor=index_torch, src=0)
        val_index = index_torch.detach().cpu().numpy()
    dist.barrier()

    train_dataset.random_split(train_index)
    val_dataset.random_split(val_index)
    
    
    trainer = training.Trainer(ddp_model, ddp_model.device, train_dataset, val_dataset,exp_name, rank = rank, world_size = world_size, optimizer=args.optimizer)
    dist.barrier()
    trainer.train_model(1500)

    cleanup()

    

if __name__ == '__main__':
    # python train.py -posed -dist 0.5 0.5 -std_dev 0.15 0.05 -res 32 -batch_size 40 -m
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    parser = argparse.ArgumentParser(
        description='Run Model'
    )


    parser.add_argument('-pointcloud', dest='pointcloud', action='store_true')
    parser.add_argument('-voxels', dest='pointcloud', action='store_false')
    parser.set_defaults(pointcloud=False)
    parser.add_argument('-pc_samples' , default=3000, type=int)
    parser.add_argument('-dist','--sample_distribution', default=[0.5, 0.5], nargs='+', type=float)
    parser.add_argument('-std_dev','--sample_sigmas',default=[0.15,0.015], nargs='+', type=float)
    parser.add_argument('-batch_size' , default=30, type=int)
    parser.add_argument('-res' , default=32, type=int)
    parser.add_argument('-m','--model' , default='LocNet', type=str)
    parser.add_argument('-o','--optimizer' , default='Adam', type=str)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_known_args()[0]


    



    
    exp_name = 'i{}_dist-{}sigmas-{}v{}_m{}'.format(  'PC' + str(args.pc_samples) if args.pointcloud else 'Voxels',
                                        ''.join(str(e)+'_' for e in args.sample_distribution),
                                        ''.join(str(e) +'_'for e in args.sample_sigmas),
                                                                    args.res,args.model)


    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus


    processes = []

    random.seed(time.time())

    #net = None
    #args = None
    
    mp.spawn(train_basic,
             args=(exp_name, world_size, args),
             nprocs=world_size,
             join=True)
    print("main finished")
    # for rank in range(world_size):
        
    #     #print(os.environ['CUDA_VISIBLE_DEVICES'])
    #     p = mp.Process(target=train_basic, args=(net, exp_name, world_size, args, train_index[:train_partial_length], val_index[:val_partial_length]))
    #     train_index = train_index[train_partial_length:]
    #     val_index = val_index[val_partial_length:]
    #     p.start()
    #     processes.append(p)

    # for p in processes:
    #     p.join()
