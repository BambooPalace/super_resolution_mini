import glob
from PIL import Image
import argparse
import os
import os.path as osp
from datetime import datetime

import mmcv
from mmcv import Config
from mmcv.runner import set_random_seed
from mmcv.runner import init_dist

import mmedit
from mmedit.datasets import build_dataset
from mmedit.models import build_model
from mmedit.apis import train_model


def parser():
    parser = argparse.ArgumentParser(description='super resolution training for 500 mini DIV2K images')
    parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
    parser.add_argument('--checkpoint', default = 'latest.pth', type=str, help='checkpoint path to resume from')
    parser.add_argument('--annotate', action='store_true', help='re annotate training files if needed')
    parser.add_argument('--iter', default=10000, type=int, help='no of iterations, default 100k, pretrained uses 300k')
    parser.add_argument('--bs', default=16, type=int, help='samples per gpu, default 16')
    parser.add_argument('--worker', default=6, type=int, help='default = 6')
    parser.add_argument('--log_eva_interval', default=1000, type=int, help='no of iters per log and evaluation' )
    parser.add_argument('--checkpoint_interval', default=200, type=int, help='no of iters to save the checkpoint')
    parser.add_argument('--lr_step', action='store_true', help='change lr policy from default Cosine Restart to step')
    parser.add_argument('--work_dir', default='./results/',type=str, help='path to save checkpoint and logs')
    parser.add_argument('--num_blocks', default=16, type=int, help='default 16 blocks in srresnet')
    parser.add_argument('--loss', default='L1Loss', type=str, choices=['L1Loss', 'MSELoss'], help='set loss function')
    parser.add_argument('--check_param', action='store_true', help='only check no of parameters')
    args = parser.parse_args()
    return args

args = parser()

def train_annotation():
    gt_paths = sorted(glob.glob('./data/Mini-DIV2K/Train/HR/*.png'))
    with open('data/training_ann.txt', 'w') as f:
      for gt_path in gt_paths:
        filename = gt_path.split('/')[-1]
        #check image size
        img = Image.open(gt_path)
        w, h = img.size
        line = f'{filename} ({w},{h},3)\n'
        f.write(line)

def change_config(config_path):

    cfg = Config.fromfile(config_path)
    # print(cfg.pretty_text)

    #set model parameters
    cfg.model.generator.num_blocks = args.num_blocks
    cfg.model.pixel_loss.type = args.loss

    # Training folders
    cfg.data.train.dataset.lq_folder = './data/Mini-DIV2K/Train/LR_x4/'
    cfg.data.train.dataset.gt_folder = './data/Mini-DIV2K/Train/HR/'
    cfg.data.train.dataset.ann_file = './data/training_ann.txt'

    # Validation folders
    cfg.data.val.lq_folder = './data/Mini-DIV2K/Val/LR_x4/'
    cfg.data.val.gt_folder = './data/Mini-DIV2K/Val/HR/'

    # TEST folders #which folder is used for calculating metrics?
    cfg.data.test.lq_folder = './data/test/'
    cfg.data.test.gt_folder = './data/fake/'


    #Resume from checkpoint
    if args.resume or args.inference:
        cfg.resume_from = args.work_dir + args.checkpoint
    else:
        cfg.resume_from = None

    # Set up working dir to save files and logs
    cfg.work_dir = args.work_dir

    # Use smaller batch size for training
    cfg.data.samples_per_gpu = args.bs #DEFAULT 16 takes 1hr/1k iter
    cfg.data.workers_per_gpu = args.worker
    cfg.data.val_workers_per_gpu = args.worker

    # Reduce the number of iterations
    cfg.total_iters = args.iter

    # Training scheme change to step
    if args.lr_step:
        cfg.lr_config = {}
        cfg.lr_config.policy = 'Step'
        cfg.lr_config.by_epoch = False
        cfg.lr_config.step = [args.iter/2]
        cfg.lr_config.gamma = 0.5
    else:
        #update lr cosinestart schedule
        period = 300000
        cfg.lr_config.periods = [period,period,period,period]
        cfg.lr_config.restart_weights = [0.1, 0.1, 0.1, 0.1]
        cfg.lr_config.min_lr = 1e-06

    # Evaluate every 1000 iterations
    cfg.evaluation.interval = args.log_eva_interval
    if cfg.evaluation.get('gpu_collect', None):
      cfg.evaluation.pop('gpu_collect')

    # Save the checkpoints every N iterations
    cfg.checkpoint_config.interval = args.checkpoint_interval

    # Print out the log every N iterations
    cfg.log_config.interval = args.log_eva_interval

    # Set seed thus the results are reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpus = 1

    cfg.dump('restoration_config.py')
    return cfg

def check_params(cfg):

    model = build_model(cfg.model).cuda()
#     print(model)
    num_param = sum(p.numel() for p in model.parameters())
    if num_param > 1821085:
        raise Exception('model parameters exceed limit')
    else:
        print('there are total of {} parameters in the model'.format(num_param))


def main():

    print('settings:\n', args)
    #annotate training data for 1st time
    if args.annotate:
        train_annotation()

    #change config
    config_path = 'configs/restorers/srresnet_srgan/msrresnet_x4c64b16_g1_1000k_div2k.py'
    cfg = change_config(config_path)
    check_params(cfg)

    # Initialize distributed training (only need to initialize once), comment it if have already run this part
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500' #'50297'
    init_dist('pytorch', **cfg.dist_params)


    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the SRCNN model
    model = build_model(
            cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # Meta information
    meta = dict()
    # if cfg.get('exp_name', None) is None:
    #     cfg['exp_name'] = osp.splitext(osp.basename(cfg.work_dir))[0]
    meta['exp_name'] = '_'.join(['bs'+str(args.bs), 'iter'+str(args.iter), 'block'+str(args.num_blocks), args.loss])
    meta['mmedit Version'] = mmedit.__version__
    meta['seed'] = 0
    meta['start_time'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    # Train the model
    train_model(model, datasets, cfg, distributed=True, validate=True, meta=meta)



if __name__ == '__main__':

    if args.check_param:
        config_path = 'configs/restorers/srresnet_srgan/msrresnet_x4c64b16_g1_1000k_div2k.py'
        cfg = change_config(config_path)
        check_params(cfg)
    else:
        main()
