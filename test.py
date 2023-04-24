# %%
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import glob as glob


def read_log_files(log_files):
    # Initialize variables to store AP50 and epoch
    ap50 = {}
    epoch = {}

    base_path = '/home/abadr/master/mmdetection/benchmark'
    # Iterate over each log file and extract AP50 and epoch for validation mode
    for label, expriment in log_files.items():
        file_name = max(glob.glob(os.path.join(base_path, expriment, '*.json')), key=os.path.getctime)
        with open(file_name, 'r') as f:
            lines = f.readlines()
        ap50[label] = []
        epoch[label] = []
        for line in lines:
            json_obj = json.loads(line.strip())
            if 'mode' in json_obj and json_obj['mode'] == 'val':
                epoch[label].append(json_obj['epoch'])
                ap50[label].append(json_obj['AP50'])
    return epoch, ap50


def plot_epoch_ap50(epoch, ap50, title):
    # Set seaborn style
    sns.set_style("whitegrid")

    base_path = '/home/abadr/master/snow_leopard_thesis/figures'
    # Plot with adjusted formatting and legends
    plt.figure(figsize=(12, 6))  # set figsize to (12,6)
    for label in epoch.keys():
        plt.plot(epoch[label], ap50[label], marker='o', markersize=5, linewidth=2, label=label)
    # plt.xticks(epoch[label])
    plt.xticks(range(0, max(epoch[label]) + 1, 100))
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('AP50', fontsize=14)
    plt.title(title, fontsize=16)
    plt.tick_params(labelsize=12)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(base_path, title.replace(' ', '_') + '.png'), dpi=300, bbox_inches='tight') # save plot to file
    plt.show()


# %%
log_files = {
    'lr0.01 bs24': 'leopard_YOLOV3_MobileNetV2_SGD_lr0.01_mom0.9_wd0.0001_ep1000',
    'lr0.00001 bs24': 'leopard_YOLOV3_MobileNetV2_SGD_lr0.00001_mom0.9_wd0.0001_ep1000',
    'lr0.00001 bs8': 'leopard_YOLOV3_MobileNetV2_SGD_lr0.00001_mom0.9_wd0.0001_ep1000_bs8',
}
epoch, ap50 = read_log_files(log_files)
plot_epoch_ap50(epoch, ap50, 'Fine-tuning YOLOV3 on v100')
# %%
log_files = {
    'lr0.01 bs24': 'leopard_v300_YOLOV3_MobileNetV2_SGD_lr0.01_mom0.9_wd0.0001_ep1000',
    'lr00001 bs24': 'leopard_v300_YOLOV3_MobileNetV2_SGD_lr0.00001_mom0.9_wd0.0001_ep1000',
    'lr00001 bs8': 'leopard_v300_YOLOV3_MobileNetV2_SGD_lr0.00001_mom0.9_wd0.0001_ep1000_bs8',
}
epoch, ap50 = read_log_files(log_files)
plot_epoch_ap50(epoch, ap50, 'Fine-tuning YOLOV3 on v300')
# %%
log_files = {
    'v100 lr0.02 bs24': 'leopard_RetinaNet_ResNet50_SGD_lr0.02_mom0.9_wd0.0001_ep1000',
    'v300 lr0.02 bs24': 'leopard_v300_RetinaNet_ResNet50_SGD_lr0.02_mom0.9_wd0.0001_ep1000',
}
epoch, ap50 = read_log_files(log_files)
plot_epoch_ap50(epoch, ap50, 'Fine-tuning RetinaNet')

# %%
log_files = {
    'lr0.01 bs24': 'leopard_HybridTaskCascade_ResNet50_SGD_ep10_lr_0.01_step200_250',
    'lr0.0001 bs24': 'leopard_HybridTaskCascade_ResNet50_SGD_ep50_lr0.0001_step200_250',
    'lr0.0025 8-11 bs24': 'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step8_11',
    'lr0.0025 8-50 bs24': 'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step8_50',
    'lr0.0025 50-200 bs4': 'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs4',
    'lr0.0025 50-200 bs8': 'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs8',
    'lr0.0025 50-200 bs32': 'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs32',
    'lr0.0025 50-200 bs24': 'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200',
    'lr0.0025 150-200 bs24': 'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step150_200',
}
epoch, ap50 = read_log_files(log_files)
plot_epoch_ap50(epoch, ap50, 'Fine-tuning HybridTaskCascade on v100')

# %%
log_files = {
    'lr0.01 bs24': 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep10_lr_0.01_step200_250',
    'lr0.0001 bs24': 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep50_lr0.0001_step200_250',
    'lr0.0025 8-11 bs24': 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step8_11',
    'lr0.0025 8-50 bs24': 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step8_50',
    'lr0.0025 50-200 bs4': 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs4',
    'lr0.0025 50-200 bs8': 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs8',
    'lr0.0025 50-200 bs32': 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs32',
    'lr0.0025 50-200 bs24': 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200',
    'lr0.0025 150-200 bs24': 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step150_200',
}
epoch, ap50 = read_log_files(log_files)
plot_epoch_ap50(epoch, ap50, 'Fine-tuning HybridTaskCascade on v300')

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
