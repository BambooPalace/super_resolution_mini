exp_name = 'msrresnet_x4c64b16_g1_1000k_div2k'
scale = 4
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='MSRResNet',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=20,
        upscale_factor=4),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=4)
train_dataset_type = 'SRAnnotationDataset'
val_dataset_type = 'SRFolderDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='PairedRandomCrop', gt_patch_size=128),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='unchanged'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='unchanged'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq', 'gt'],
        mean=[0, 0, 0],
        std=[1, 1, 1],
        to_rgb=True),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'lq_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=6,
    drop_last=True,
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type='SRAnnotationDataset',
            lq_folder='./data/Mini-DIV2K/Train/LR_x4_sub/',
            gt_folder='./data/Mini-DIV2K/Train/HR_sub/',
            ann_file='./data/training_ann.txt',
            pipeline=[
                dict(
                    type='LoadImageFromFile',
                    io_backend='disk',
                    key='lq',
                    flag='unchanged'),
                dict(
                    type='LoadImageFromFile',
                    io_backend='disk',
                    key='gt',
                    flag='unchanged'),
                dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
                dict(
                    type='Normalize',
                    keys=['lq', 'gt'],
                    mean=[0, 0, 0],
                    std=[1, 1, 1],
                    to_rgb=True),
                dict(type='PairedRandomCrop', gt_patch_size=128),
                dict(
                    type='Flip',
                    keys=['lq', 'gt'],
                    flip_ratio=0.5,
                    direction='horizontal'),
                dict(
                    type='Flip',
                    keys=['lq', 'gt'],
                    flip_ratio=0.5,
                    direction='vertical'),
                dict(
                    type='RandomTransposeHW',
                    keys=['lq', 'gt'],
                    transpose_ratio=0.5),
                dict(
                    type='Collect',
                    keys=['lq', 'gt'],
                    meta_keys=['lq_path', 'gt_path']),
                dict(type='ImageToTensor', keys=['lq', 'gt'])
            ],
            scale=4)),
    val_samples_per_gpu=1,
    val_workers_per_gpu=6,
    val=dict(
        type='SRFolderDataset',
        lq_folder='./data/Mini-DIV2K/Val/LR_x4/',
        gt_folder='./data/Mini-DIV2K/Val/HR/',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='lq',
                flag='unchanged'),
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='gt',
                flag='unchanged'),
            dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
            dict(
                type='Normalize',
                keys=['lq', 'gt'],
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True),
            dict(
                type='Collect',
                keys=['lq', 'gt'],
                meta_keys=['lq_path', 'lq_path']),
            dict(type='ImageToTensor', keys=['lq', 'gt'])
        ],
        scale=4,
        filename_tmpl='{}'),
    test=dict(
        type='SRFolderDataset',
        lq_folder='./data/test/',
        gt_folder='./data/fake/',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='lq',
                flag='unchanged'),
            dict(
                type='LoadImageFromFile',
                io_backend='disk',
                key='gt',
                flag='unchanged'),
            dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
            dict(
                type='Normalize',
                keys=['lq', 'gt'],
                mean=[0, 0, 0],
                std=[1, 1, 1],
                to_rgb=True),
            dict(
                type='Collect',
                keys=['lq', 'gt'],
                meta_keys=['lq_path', 'lq_path']),
            dict(type='ImageToTensor', keys=['lq', 'gt'])
        ],
        scale=4,
        filename_tmpl='{}'))
optimizers = dict(generator=dict(type='Adam', lr=0.0002, betas=(0.9, 0.999)))
total_iters = 1000000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[250000, 250000, 250000, 250000],
    restart_weights=[1, 1, 1, 1],
    min_lr=1e-07)
checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=10000, save_image=True)
log_config = dict(
    interval=10000,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
visual_config = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './results/'
load_from = None
resume_from = './results/latest.pth'
workflow = [('train', 1)]
seed = 0
gpus = 1
