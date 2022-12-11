# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import json

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
                 model_name,
                 tb=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name 
        self.metric = metric
        self.tb = tb

    def train(self, epoch, args):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        time_sync(args.device)
        data_time = 0.0
        total_forward_time = 0.0
        total_backward_time = 0.0
        total_loss = 0.0
        data_count = 0
        epoch_start_time = time.time()
        end = epoch_start_time

        for batch_id, (images, target) in enumerate(self.train_loader):            
            # move data to the same device as model
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            data_count += images.size(0)
            # measure data loading time
            time_sync(args.device)
            data_time += (time.time() - end)
            
            time_sync(args.device)
            forward_time = time.time()
            # compute output
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(): 
                output = self.model(images)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
            time_sync(args.device)
            total_forward_time += (time.time() - forward_time)

            # compute gradient and do SGD step
            time_sync(args.device)
            backward_time = time.time()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.zero_grad(set_to_none=True)

            time_sync(args.device)
            total_backward_time += (time.time() - backward_time)

            time_sync(args.device)
            end = time.time()

        time_sync(args.device)
        self.tb.add_scalar("Loss", total_loss, epoch)
        self.tb.add_scalar("Data Load Time", data_time, epoch)
        self.tb.add_scalar("Data Throughput", (data_count/data_time), epoch)
        self.tb.add_scalar("Forward Time", total_forward_time, epoch)
        self.tb.add_scalar("Backward Time", total_backward_time, epoch)
        
        metrics = {
            'epoch': epoch,
            'dataset_size': batch_id,
            'data_time': round(data_time, 3),
            'data_throughput': round((data_count/data_time), 0),
            'forward_time': round(total_forward_time, 3),
            'backward_time': round(total_backward_time, 3)
        }
        metrics_fmt = ', '.join([f'{k}={v}' for k,v in metrics.items()])
        print(f'=> [Epoch {epoch}] [{args.gpu}]: {metrics_fmt}')
        return metrics


    def validate(self, epoch, args):
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
        self.tb.add_scalar("Val Accuracy", accuracy, epoch)
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

            acc1 = self.validate(epoch, args)

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

        # write metrics for visualization
        get_metric = lambda m,key: [e[key] for e in m]
        avg = lambda items: sum(items)/len(items)

        m_epoch = train_metrics['epoch']
        m_total = train_metrics['total']
        m_total['train_time'] = round(total_train_time/args.epochs, 3)
        m_total['eval_time']  = round(total_eval_time/args.epochs, 3)
        m_total['data_load_time'] = round(avg(get_metric(m_epoch, 'data_time')), 3)
        m_total['data_throughput'] = round(avg(get_metric(m_epoch, 'data_throughput')), 3)
        m_total['forward_time'] = round(avg(get_metric(m_epoch, 'forward_time')), 3)
        m_total['backward_time'] = round(avg(get_metric(m_epoch, 'backward_time')), 3)

        # m_total['avg_training_loss'] = round(sum(get_metric(m_epoch, 'loss'))/len(m_epoch), 4)

        metrics_path = os.path.join(args.metrics_dir, f'metrics_{args.gpu}.json')
        print(f'Writing metrics to {metrics_path}')
        with open(metrics_path, 'w') as f_metrics:
            json.dump(train_metrics, f_metrics)