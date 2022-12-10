import os
import time

import torch


def time_sync(device):
    if device != 'cpu':
        torch.cuda.synchronize()


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

        time_sync(args.device)
        data_time = 0.0
        train_time = 0.0
        total_loss = 0.0
        data_count = 0
        batch_count = 0
        epoch_start_time = time.time()
        end = epoch_start_time

        for i, (images, target) in enumerate(self.train_loader):
            # measure data loading time
            time_sync(args.device)
            data_time += (time.time() - end)
            
            # move data to the same device as model
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            data_count += images.size(0)
            # compute output
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(): 
                output = self.model(images)
                loss = self.criterion(output, target)
            total_loss += (loss.item() * images.size(0))
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.zero_grad(set_to_none=True)
            
            batch_count += 1
            time_sync(args.device)
            end = time.time()

        time_sync(args.device)
        train_time = (time.time() - epoch_start_time)
        metrics = {
            'epoch': epoch,
            'dataset_size': batch_count,
            'data_time': round(data_time, 3),
            'data_throughput': round((data_count/data_time), 0),
            # 'loss': round(total_loss/batch_count, 4),
            'train_time': round(train_time, 3)
        }
        metrics_fmt = ', '.join([f'{k}={v}' for k,v in metrics.items()])
        print(f'=> [Epoch {epoch}] [{args.gpu}]: {metrics_fmt}')
        return metrics


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
            print(f"=> Accuracy: {round(accuracy.item(), 6)}")
        return accuracy


    def save_checkpoint(self, state, filename='checkpoint.pt'):
        torch.save(state, filename)


    def run(self, args):
        train_metrics = {'gpu': args.gpu,
                         'total': {}, 
                         'epoch': []}
        total_train_time = 0.0
        total_eval_time = 0.0

        # training loop begin
        for epoch in range(1, args.epochs + 1):
            if args.distributed and not args.webdataset:
                self.train_loader.sampler.set_epoch(epoch)

            if args.gpu == 0:
                print(f'=> [Epoch {epoch}]: Starting')

            # train for one epoch
            time_sync(args.device)
            start = time.time()

            epoch_metrics = self.train(epoch, args)
            
            train_metrics['epoch'].append(epoch_metrics)
            time_sync(args.device)
            end = time.time()
            train_time = (end - start)
            total_train_time += train_time
            if args.gpu == 0:
                print(f'=> [Epoch {epoch}]: Training finished in {train_time:>0.3f} seconds')

            # evaluate on validation set
            time_sync(args.device)
            start = time.time()
            acc1 = self.validate(args)
            time_sync(args.device)
            end = time.time()
            eval_time = (end - start)
            total_eval_time += eval_time
            if args.gpu == 0:
                print(f'=> [Epoch {epoch}]:Evaluation finished in {eval_time:>0.3f} seconds')
                
            # save checkpoint
            if args.gpu == 0 and args.distributed_strategy != 'fsdp':
                print(f'=> Saving checkpoint to {self.model_name}.pt')
                # if args.distributed_strategy == 'fsdp':
                #     from torch.distributed.fsdp import (
                #         FullyShardedDataParallel as FSDP,
                #         FullStateDictConfig,
                #         StateDictType
                #     )
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
                }, filename=os.path.join(args.model_dir, f'{self.model_name}.pt'))
        # training loop end

        get_metric = lambda m,key: [e[key] for e in m]
        epoch_meter = train_metrics['epoch']
        total_meter = train_metrics['total']
        total_meter['total_train_time'] = round(total_train_time, 3)
        total_meter['avg_epoch_train_time'] = round(sum(get_metric(epoch_meter, 'train_time'))/len(epoch_meter), 3)
        total_meter['total_eval_time']  = round(total_eval_time, 3)
        total_meter['total_data_load_time'] = round(sum(get_metric(epoch_meter, 'data_time'))/len(epoch_meter), 3)
        total_meter['data_throughput'] = sum(get_metric(epoch_meter, 'data_throughput'))
        # total_meter['avg_training_loss'] = round(sum(get_metric(epoch_meter, 'loss'))/len(epoch_meter), 4)
        # if args.gpu == 0:
        #     metrics_fmt = '\n=> '.join([f'{k}={v}' for k,v in total_meter.items()])
        #     print('-'*80)
        #     print(f'=> {metrics_fmt}')
        #     print('-'*80)