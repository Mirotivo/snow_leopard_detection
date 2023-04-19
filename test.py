# %%
import glob
import mmcv
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.datasets import build_dataset


# expriment = 'balloon_HybridTaskCascade_ResNet50_SGD_ep10_lr_0.01_step200_250'
expriment = 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs4'
config_file = f'mmdetection/configs/amr/{expriment}.py'
checkpoint_file = f'mmdetection/benchmark/{expriment}/best_mAP_epoch_*.pth'
files = glob.glob(checkpoint_file)
checkpoint_file = files[0]
model = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'


# Test Image
# img = 'mmdetection/data/balloon_VOC2012/JPEGImages/53500107_d24b11b3c2_b.jpg'
# result = inference_detector(model, img)
# show_result_pyplot(model, img, result, score_thr=0.3)

cfg = mmcv.Config.fromfile(config_file)
cfg.data.val.data_root = 'mmdetection/' + cfg.data.val.data_root
dataset = build_dataset(cfg.data.val)
# img_names = [img_info['filename'] for img_info in img_infos]
img_names = [info['filename'] for info in dataset.data_infos]
for img_name in img_names:
    img = mmcv.imread(f'{cfg.data.val.data_root}/{img_name}')
    result = inference_detector(model, img)
    show_result_pyplot(model, img, result, score_thr=0.5)

# %%
