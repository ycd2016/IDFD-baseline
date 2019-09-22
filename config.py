# @Author: Chauncy Yao
'''
MMDetection: Open MMLab Detection Toolbox and Benchmark
https://arxiv.org/pdf/1906.07155.pdf
'''
# 模型配置
'''
Group Normalization
https://arxiv.org/pdf/1803.08494.pdf
'''
norm_cfg = dict(type='GN', num_groups=64, requires_grad=True)
'''
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
https://arxiv.org/pdf/1506.01497.pdf
'''
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        style='pytorch',
        norm_cfg=norm_cfg,
'''
Deformable ConvNets v2: More Deformable, Better Results
https://arxiv.org/pdf/1811.11168.pdf
'''
        dcn=dict( # 加入可变形卷积
            groups=64,
            deformable_groups=1,
            fallback_on_stride=False),
        stage_with_dcn=(True, True, True, True)),
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5,
            norm_cfg=norm_cfg),
'''
Libra R-CNN: Towards Balanced Learning for Object Detection
https://arxiv.org/pdf/1904.02701.pdf
'''
        dict(
            type='BFP',
            in_channels=256,
            num_levels=5,
            refine_level=2,
            refine_type='non_local')
    ],
'''
Region Proposal by Guided Anchoring
https://arxiv.org/pdf/1901.03278.pdf
'''
    rpn_head=dict(
        type='GARPNHead',
        in_channels=256,
        feat_channels=256,
        octave_base_scale=8,
        scales_per_octave=3,
        octave_ratios=[0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 10.0, 20.0, 50.0], # 修改 octave 长宽比
        anchor_strides=[4, 8, 16, 32, 64],
        anchor_base_sizes=None,
        anchoring_means=[.0, .0, .0, .0],
        anchoring_stds=[0.07, 0.07, 0.14, 0.14],
        target_means=(.0, .0, .0, .0),
        target_stds=[0.07, 0.07, 0.11, 0.11],
        loc_filter_thr=0.01,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0), # 此处不要用 Focalloss
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=21, # 20 种瑕疵 + 背景
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(
            type='BalancedL1Loss',
            alpha=0.5,
            gamma=1.5,
            beta=1.0,
            loss_weight=1.0)))
# 训练与推理配置
train_cfg = dict(
    rpn=dict(
        ga_assigner=dict(
            type='ApproxMaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        ga_sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=-1,
        pos_weight=-1,
        center_ratio=0.2,
        ignore_ratio=0.5,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2200,
        nms_post=2200,
        max_num=300,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict( # 参考上面的 Libra R-CNN
            type='CombinedSampler',
            num=512,
            pos_fraction=0.25,
            add_gt_as_proposals=True,
            pos_sampler=dict(type='InstanceBalancedPosSampler'),
            neg_sampler=dict(
                type='IoUBalancedNegSampler',
                floor_thr=-1,
                floor_fraction=0,
                num_bins=3)),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1100,
        nms_post=1100,
        max_num=300,
        nms_thr=0.7,
        min_bbox_size=0),
'''
Soft-NMS -- Improving Object Detection With One Line of Code
https://arxiv.org/pdf/1704.04503.pdf
'''
    rcnn=dict(
        score_thr=0.08, nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.08),
        max_per_img=100)
)
# 数据集配置
dataset_type = 'CocoDataset'
data_root = '../data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(1216, 512), (1408, 576)], keep_ratio=True), # 显存不够只好缩小尺寸，多尺度训练
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1216, 512), (1408, 576), (1600, 640)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=1, # 显存不够只好 batch_size = 1
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train.json', # 训练集标注文件
        img_prefix=data_root + '../data/train/', # 训练集图片
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'valid.json', # 验证集标注文件
        img_prefix=data_root + '../data/valid/', # 验证集图片
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'testb.json', # 测试集标注文件
        img_prefix=data_root + '../data/testb/', # 测试集图片
        pipeline=test_pipeline))
# 优化器
optimizer = dict(type='SGD', lr=0.022, momentum=0.9, weight_decay=0.0001) # 玄学调参~
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# 学习率策略
lr_config = dict(
    policy='cosine',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TensorboardLoggerHook')
    ])
# 运行时配置
total_epochs = 24
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '../data/work_dir' # 权重和日志保存路径
load_from = None # 如需加载预训练权重
resume_from = None
workflow = [('train', 1)]
