import time

import torch

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType
)

class Trainer():

    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 train_loader,
                 val_loader,
                 metric,
                 model_name):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name 
        self.metric = metric

    def train(self, epoch, args):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for i, (images, target) in enumerate(self.train_loader):
            # move data to the same device as model
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            # compute output
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(): 
                output = self.model(images)
                loss = self.criterion(output, target)
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.zero_grad(set_to_none=True)


    def validate(self, args):
        # switch to evaluate mode
        self.model.eval()
        with torch.no_grad():
            for (images, target) in self.val_loader:
                # move data to the same device as model
                images = images.to(args.device, non_blocking=True)
                target = target.to(args.device, non_blocking=True)
                # compute output
                output = self.model(images)
                self.metric.update(output, target)

        accuracy = self.metric.compute()
        self.metric.reset()
        
        if args.gpu == 0:
            print(f"=> Accuracy: {accuracy}")
        return accuracy


    def save_checkpoint(self, state, filename='checkpoint.pt'):
        torch.save(state, filename)


    def run(self, args):
        for epoch in range(1, args.epochs + 1):
            if args.distributed and not args.webdataset:
                self.train_loader.sampler.set_epoch(epoch)

            if args.gpu == 0:
                print(f'=> [Epoch {epoch}]: Starting')

            # train for one epoch
            start = time.time()
            self.train(epoch, args)
            end = time.time()
            if args.gpu == 0:
                print(f'=> [Epoch {epoch}]: Training finished in {(end - start):>0.3f} seconds')

            # evaluate on validation set
            start = time.time()
            acc1 = self.validate(args)
            end = time.time()
            if args.gpu == 0:
                print(f'=> [Epoch {epoch}]:Evaluation finished in {(end - start):>0.3f} seconds')
                
            # save checkpoint
            if args.gpu == 0 and args.distributed_strategy != 'fsdp':
                print(f'=> Saving checkpoint to {self.model_name}.pt')
                # if args.distributed_strategy == 'fsdp':
                #     save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                #     with FSDP.state_dict_type(
                #         self.model, StateDictType.FULL_STATE_DICT, save_policy):
                #         state_dict = self.model.state_dict()
                # else:
                state_dict = self.model.state_dict()

                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': state_dict,
                    'optimizer' : self.optimizer.state_dict()
                }, filename=f"{self.model_name}.pt")