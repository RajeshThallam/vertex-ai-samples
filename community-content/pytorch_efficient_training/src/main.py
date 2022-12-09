import argparse
import os
import time
import functools

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torchvision.models import resnet50
import torchmetrics

from src import dataset, experiment


def parse_args():
    """Create main args."""
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--gpus', default=4, type=int,
                        help='number of gpus to use')
    parser.add_argument('--train_data_path', default='', type=str,
                        help='path to training dataset. For Webdataset, path to set as /path/to/filename-{000000..001000}.tar')
    parser.add_argument('--val_data_path', default='', type=str,
                        help='path to validation dataset. For Webdataset, path to set as /path/to/filename-{000000..001000}.tar')
    parser.add_argument('--data-size', default=50000, type=int, 
                        help='data size for training')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run (default: 1)')
    parser.add_argument('--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32), this is the '
                             'batch size per gpu on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--webdataset', action='store_true', 
                        help='use webdataset data loader (default: False)')
    parser.add_argument('--distributed-strategy', default='ddp', type=str, 
                        help='Training distribution strategy. Valid values are: dp, ddp, fsdp')
    # to run the job using torchrun
    parser.add_argument("--hostip", default="localhost", type=str, 
                        help="setting for etcd host ip")
    parser.add_argument("--hostipport", default=2379, type=int, 
                        help="setting for etcd host ip port",)
    args = parser.parse_args()
    return args


def init_ddp(args):
    # init process group
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.gpus,
                            rank=args.gpu)

def set_env(args):
    env_defaults = {
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': '8888',
        'WORLD_SIZE': str(args.gpus), # of gpus on a single node
    }
    for key, value in env_defaults.items():
        if key not in os.environ:
            os.environ[key] = value


def prepare_model(args):
    # create model
    if args.gpu == 0:
        print("=> creating model resnet50")
    args.arch = 'resnet50'
    model = resnet50(weights=None)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        model_name = f'{args.arch}-{args.device}{"-wds" if args.webdataset else ""}'
    else:
        torch.cuda.set_device(args.gpu)
        model.to(args.device)
        if args.distributed_strategy == 'ddp':
            model = DDP(model)
        elif args.distributed_strategy == 'fsdp':
            fsdp_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
            model = FSDP(model, auto_wrap_policy=fsdp_auto_wrap_policy)
        elif args.distributed_strategy == 'dp':
             model = DP(model)

        model_name = f'resnet50-{args.device}-{args.distributed_strategy}{"-wds" if args.webdataset else ""}'

    return model, model_name


def main():
    args = parse_args()    
    set_env(args)

    if torch.cuda.is_available():
        args.device = 'cuda' if args.distributed_strategy in ['ddp', 'fsdp'] else 'cuda:0'
        args.distributed = args.gpus > 1
    else: 
        args.device = 'cpu'
        args.distributed = False

    if args.distributed:
        print(f'Launch job on {args.gpus} GPUs with {args.distributed_strategy}')
        mp.spawn(main_worker, nprocs=args.gpus, args=(args,))
    else:
        main_worker(0, args)


def main_worker(gpu, args):
    args.gpu = gpu
    if torch.cuda.is_available() and args.gpu is not None:
        print(f"=> Use GPU: {args.gpu} for training")
    else:
        print(f"=> Use CPU for training")

    if args.distributed and args.distributed_strategy in ('ddp', 'fsdp'):
        init_ddp(args)

    # prepare model
    model, model_name = prepare_model(args)
    args.model_name = model_name 
    if args.gpu == 0:
        print(f"=> Model name = {model_name}")

    # define loss function (criterion), optimizer
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr)
    metric = torchmetrics.classification.Accuracy(top_k=1).to(args.device)

    # prepare dataloader
    if args.webdataset:
        train_loader = dataset.prepare_wds_dataloader('train', args)
        val_loader = dataset.prepare_wds_dataloader('val', args)
    else:
        train_loader = dataset.prepare_dataloader('train', args)
        val_loader = dataset.prepare_dataloader('val', args)

    # training loop
    trainer = experiment.Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        metric=metric,
        model_name=model_name
    )

    if args.gpu == 0:
        for arg in vars(args):
            print(f'{arg} = {getattr(args, arg)}')
    
    if args.evaluate:
        trainer.validate(args)
        return

    trainer.run(args)

if __name__ == '__main__':
    main()