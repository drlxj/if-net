from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import torch.distributed as dist



class Trainer(object):

    def __init__(self, model, device, train_dataset, val_dataset, exp_name, rank, world_size, optimizer='Adam', parallel = False):
        # self.model = model.to(device)
        self.model = model
        self.device = device
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters())
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format( exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format( exp_name)
        if not os.path.exists(self.checkpoint_path):
            if not parallel or not rank:
                print(self.checkpoint_path)
                os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
        self.val_min = None
        self.rank = rank
        self.world_size = world_size
        self.parallel = parallel


    def train_step(self,batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_loss(self,batch):
        device = self.device

        p = batch.get('grid_coords').to(device)
        occ = batch.get('occupancies').to(device)
        inputs = batch.get('inputs').to(device)


        # General points
        logits = self.model(p,inputs)
        loss_i = F.binary_cross_entropy_with_logits(
            logits, occ, reduction='none')# out = (B,num_points) by componentwise comparing vecots of size num_samples:
        # l(logits[n],occ[n]) for each n in B. i.e. l(logits[n],occ[n]) is vector of size num_points again.

        loss = loss_i.sum(-1).mean() # loss_i summed over all #num_samples samples -> out = (B,1) and mean over batch -> out = (1)

        return loss

    def train_model(self, epochs):
        loss = 0
        # start = self.load_checkpoint()
        map_location = f'cuda:{self.rank}'
        start = self.load_checkpoint(map_location=map_location)

        for epoch in range(start, epochs):
            sum_loss = 0
            print('Start epoch {}'.format(epoch))
            train_data_loader = self.train_dataset.get_loader()

            if epoch % 1 == 0:
                if self.parallel :
                    if not self.rank:
                        self.save_checkpoint(epoch)
                    dist.barrier()
                    val_loss_from_others = torch.zeros(1)
                    val_loss = self.compute_val_loss().cpu()
                    if not self.rank:
                        for rank in range(1,self.world_size):
                            dist.recv(tensor=val_loss_from_others, src=rank)
                            val_loss += val_loss_from_others
                        val_loss = val_loss/self.world_size
                    else:
                        dist.send(tensor=val_loss, dst=0)
                    dist.barrier()
                    if not self.rank:
                        if self.val_min is None:
                            self.val_min = val_loss

                            if val_loss < self.val_min:
                                self.val_min = val_loss
                                for path in glob(self.exp_path + 'val_min=*'):
                                    os.remove(path)
                                np.save(self.exp_path + 'val_min={}'.format(epoch),[epoch,val_loss])


                            self.writer.add_scalar('val loss batch avg', val_loss, epoch)
                else:
                    self.save_checkpoint(epoch)
                    val_loss = self.compute_val_loss()

                    if self.val_min is None:
                        self.val_min = val_loss

                    if val_loss < self.val_min:
                        self.val_min = val_loss
                        for path in glob(self.exp_path + 'val_min=*'):
                            os.remove(path)
                        np.save(self.exp_path + 'val_min={}'.format(epoch),[epoch,val_loss])


                    self.writer.add_scalar('val loss batch avg', val_loss, epoch)


            for ib, batch in enumerate(train_data_loader):
                loss = self.train_step(batch)
                print("epoch: {}, batch: {}, Current loss: {}".format(epoch, ib, loss))
                sum_loss += loss


            self.writer.add_scalar('training loss last batch', loss, epoch)
            self.writer.add_scalar('training loss batch avg', sum_loss / len(train_data_loader), epoch)



    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):
            torch.save({'epoch':epoch,'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path)

    def load_checkpoint(self, map_location):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        return epoch

    def compute_val_loss(self):
        self.model.eval()

        sum_val_loss = 0
        num_batches = 15
        for _ in range(num_batches):
            try:
                val_batch = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_dataset.get_loader().__iter__()
                val_batch = self.val_data_iterator.next()

            sum_val_loss += self.compute_loss( val_batch).item()

        return sum_val_loss / num_batches
