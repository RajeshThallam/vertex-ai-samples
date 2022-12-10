import argparse
import os
import time
import functools
import subprocess
import shutil

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet50
import torchmetrics

from trainer import dataset, experiment


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
    parser.add_argument('--local_disk', action='store_true', 
                        help='download train and eval data to local disk  (default: False)')
    parser.add_argument('--distributed-strategy', default='ddp', type=str, 
                        help='Training distribution strategy. Valid values are: dp, ddp, fsdp')
    # to run the job using torchrun
    parser.add_argument("--hostip", default="localhost", type=str, 
                        help="setting for etcd host ip")
    parser.add_argument("--hostipport", default=2379, type=int, 
                        help="setting for etcd host ip port",)

    # Using environment variables for Cloud Storage directories
    # see more details in https://cloud.google.com/vertex-ai/docs/training/code-requirements
    parser.add_argument('--model-dir', default=os.getenv('AIP_MODEL_DIR'), type=str, help='Cloud Storage URI to write model artifacts')
    parser.add_argument('--tensorboard-log-dir', default=os.getenv('AIP_TENSORBOARD_LOG_DIR'), type=str, help='Cloud Storage URI to write to TensorBoard')
    parser.add_argument('--checkpoint-dir', default=os.getenv('AIP_CHECKPOINT_DIR'), type=str, help='Cloud Storage URI to save checkpoints')

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

def makedirs(_dir):
    if os.path.exists(_dir) and os.path.isdir(_dir):
        shutil.rmtree(_dir)
    os.makedirs(_dir)
    return


def get_dir(_dir, local_dir):
    gs_prefix = 'gs://'
    gcsfuse_prefix = '/gcs/'
    local_dir = './tmp/model'
    _dir = _dir or local_dir
    if _dir and _dir.startswith(gs_prefix):
        _dir = _dir.replace(gs_prefix, gcsfuse_prefix)
    makedirs(_dir)
    return _dir


def prepare_model(args):
    # create model
    args.arch = 'resnet50'
    if args.gpu == 0:
        print(f"=> creating model {args.arch}")

    model = resnet50(weights=None)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        model_name = f'{args.arch}-{args.device}{"-wds" if args.webdataset else ""}'
    else:
        torch.cuda.set_device(args.gpu)
        model.to(args.device)

        if args.distributed:
            if args.distributed_strategy == 'ddp':
                model = DDP(model)
            elif args.distributed_strategy == 'fsdp':
                fsdp_auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100)
                model = FSDP(model, auto_wrap_policy=fsdp_auto_wrap_policy)
            elif args.distributed_strategy == 'dp':
                 model = DP(model)
            model_name = f'{args.arch}-{args.device}-{args.distributed_strategy}{"-wds" if args.webdataset else ""}'
        else:
            model_name = f'{args.arch}-{args.device}{"-wds" if args.webdataset else ""}'

    return model, model_name


def main():
    args = parse_args()    
    set_env(args)
    
    if torch.cuda.is_available():
        if args.gpus > torch.cuda.device_count():
            print(f'Overriding gpus to {torch.cuda.device_count()} from {args.gpus}')
            args.gpus = torch.cuda.device_count()

        args.device = 'cuda' if args.distributed_strategy in ['ddp', 'fsdp'] else 'cuda:0'
        args.distributed = args.gpus > 1
        if not args.distributed:
            args.distributed_strategy = 'n/a'
    else: 
        args.device = 'cpu'
        args.distributed = False

    if args.local_disk:
        print("Start copying data to local disk")
        subprocess.call(['sh', './copy_to_local.sh', args.train_data_path])
        subprocess.call(['sh', './copy_to_local.sh', args.val_data_path])

    args.model_dir = get_dir(args.model_dir , 'tmp/model')
    args.tensorboard_log_dir = get_dir(args.tensorboard_log_dir , 'tmp/logs')
    args.checkpoint_dir = get_dir(args.checkpoint_dir , 'tmp/checkpoints')
        
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

    tb = SummaryWriter(args.tensorboard_log_dir)
        
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

    if args.distributed:
        dist.destroy_process_group()
      
    if args.gpu == 0:
        print('Done')

if __name__ == '__main__':
    main()