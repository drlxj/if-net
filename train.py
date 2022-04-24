import models.local_model as model
import models.data.voxelized_data_shapenet as voxelized_data
from models import training
import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_basic(rank, dataset, exp_name, optimizer, world_size):
    net, train_dataset, val_dataset = dataset
    print(f"Running basic DDP on rank {rank}.")
    setup(rank, world_size)
    net = net.to(rank)
    ddp_model = DDP(net, device_ids = [rank])
    trainer = training.Trainer(ddp_model, ddp_model.device, train_dataset, val_dataset,exp_name, rank = rank, world_size = world_size, optimizer=optimizer)
    trainer.train_model(1500)

    cleanup()

    

if __name__ == '__main__':
    # python train.py -posed -dist 0.5 0.5 -std_dev 0.15 0.05 -res 32 -batch_size 40 -m
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


    if args.model ==  'ShapeNet32Vox':
        net = model.ShapeNet32Vox()

    if args.model ==  'ShapeNet128Vox':
        net = model.ShapeNet128Vox()

    if args.model == 'ShapeNetPoints':
        net = model.ShapeNetPoints()

    if args.model == 'SVR':
        net = model.SVR()



    
    exp_name = 'i{}_dist-{}sigmas-{}v{}_m{}'.format(  'PC' + str(args.pc_samples) if args.pointcloud else 'Voxels',
                                        ''.join(str(e)+'_' for e in args.sample_distribution),
                                        ''.join(str(e) +'_'for e in args.sample_sigmas),
                                                                    args.res,args.model)

    # trainer = training.Trainer(net,torch.device("cuda"),train_dataset, val_dataset,exp_name, optimizer=args.optimizer)
    # trainer.train_model(1500)

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus

    train_index_left = []
    val_index_left = []

    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        train_dataset = voxelized_data.VoxelizedDataset('train', voxelized_pointcloud= args.pointcloud, pointcloud_samples= args.pc_samples, res=args.res, sample_distribution=args.sample_distribution,
                                            sample_sigmas=args.sample_sigmas ,num_sample_points=50000, batch_size=args.batch_size, num_workers=0, world_size = world_size, rank = rank, partition_index = train_index_left)
        train_index_left = train_dataset.partition_index
        val_dataset = voxelized_data.VoxelizedDataset('val', voxelized_pointcloud= args.pointcloud , pointcloud_samples= args.pc_samples, res=args.res, sample_distribution=args.sample_distribution,
                                            sample_sigmas=args.sample_sigmas ,num_sample_points=50000, batch_size=args.batch_size, num_workers=0, world_size = world_size, rank = rank, partition_index = val_index_left)
        val_index_left = val_dataset.partition_index
        p = mp.Process(target=train_basic, args=((net, train_dataset, val_dataset), exp_name, args.optimizer, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()