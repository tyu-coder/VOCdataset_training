_base_=[
    './_base_/models/mask-rcnn_r50_fpn.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py',

]
dataset_type='CocoDataset'
data_root='data/SBD/benchmark_RELEASE/dataset/'

metainfo=dict(
    classes=(
         "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
    
    )
)
train_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations',with_bbox=True,with_mask=True),
    dict(type='Resize',scale=(1333,800),keep_ratio=True),
    dict(type='RandomFlip',prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline=[
    dict(type='LoadImageFromFile'),
    dict(type='Resize',scale=(1333,800),keep_ratio=True),
    dict(type='LoadAnnotations',with_bbox=True,with_mask=True),
    dict(type='PackDetInputs'),

]
train_dataloader=dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler',shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/sbd_train_coco.json',
        data_prefix=dict(img='img/'),
        metainfo=metainfo,
        pipeline=train_pipeline,
    ),
)
val_dataloader=dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler',shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/sbd_val_coco.json',
        data_prefix=dict(img='img/'),
        metainfo=metainfo,
        pipeline=test_pipeline,
    ),
)
test_dataloader=val_dataloader

val_evaluator=dict(
    type='CocoMetric',
    ann_file=data_root+'annotations/sbd_val_coco.json',
    metric=['bbox','segm'],
    format_only=False,
    outfile_prefix='./debug_eval/preds'

)
test_evaluator=val_evaluator

train_cfg=dict(type='EpochBasedTrainLoop',max_epochs=10,val_interval=1)
val_cfg=dict(type='ValLoop')
test_cfg=dict(type='TestLoop')

optim_wrapper=dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD',lr=0.005,momentum=0.9,weight_decay=0.0001),
)

default_hooks=dict(
    logger=dict(type='LoggerHook',interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook',interval=10,max_keep_ckpts=1),
    timer=dict(type='IterTimerHook'),
)
vis_backends=[
    dict(type='TensorboardVisBackend',save_dir='./tb_logs'),
    dict(type='LocalVisBackend',save_dir='./debug_vis')
]
visualizer=dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',

)
env_cfg=dict(device='cuda')
work_dir='./work_dirs/mask_rcnn'
model=dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(metainfo['classes'])),
        mask_head=dict(num_classes=len(metainfo['classes']))

    )
)
load_from='checkpoints/mask_rcnn_r50_fpn_1x_coco.pth'