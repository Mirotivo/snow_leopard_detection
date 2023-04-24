import subprocess

experiments = [

    # ####### YOLOV3 model ###########
    # 'balloon_YOLOV3_MobileNetV2_SGD_lr0.01_mom0.9_wd0.0001_ep1000',
    # 'balloon_YOLOV3_MobileNetV2_SGD_lr0.00001_mom0.9_wd0.0001_ep1000',
    # 'balloon_YOLOV3_MobileNetV2_SGD_lr0.00001_mom0.9_wd0.0001_ep1000_bs8',
    # 'leopard_YOLOV3_MobileNetV2_SGD_lr0.01_mom0.9_wd0.0001_ep1000',
    # 'leopard_YOLOV3_MobileNetV2_SGD_lr0.00001_mom0.9_wd0.0001_ep1000',
    # 'leopard_YOLOV3_MobileNetV2_SGD_lr0.00001_mom0.9_wd0.0001_ep1000_bs8',
    # 'leopard_v300_YOLOV3_MobileNetV2_SGD_lr0.01_mom0.9_wd0.0001_ep1000',
    # 'leopard_v300_YOLOV3_MobileNetV2_SGD_lr0.00001_mom0.9_wd0.0001_ep1000',
    # 'leopard_v300_YOLOV3_MobileNetV2_SGD_lr0.00001_mom0.9_wd0.0001_ep1000_bs8',


    # # ####### Mask RCNN model ###########



    # # ####### RetinaNet model ###########
    # 'balloon_RetinaNet_ResNet50_SGD_lr0.02_mom0.9_wd0.0001_ep1000',
    # 'leopard_RetinaNet_ResNet50_SGD_lr0.02_mom0.9_wd0.0001_ep1000',
    # 'leopard_v300_RetinaNet_ResNet50_SGD_lr0.02_mom0.9_wd0.0001_ep1000',

    # # ####### HTC model ###########
    # 'balloon_HybridTaskCascade_ResNet50_SGD_ep10_lr_0.01_step200_250',
    # 'balloon_HybridTaskCascade_ResNet50_SGD_ep50_lr0.0001_step200_250',
    # 'balloon_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step8_50',
    # 'balloon_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step8_11',
    # 'balloon_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs4',
    # 'balloon_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs8',
    # 'balloon_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs32',
    # 'balloon_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200',
    # 'balloon_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step150_200',
    # 'leopard_HybridTaskCascade_ResNet50_SGD_ep10_lr_0.01_step200_250',
    # 'leopard_HybridTaskCascade_ResNet50_SGD_ep50_lr0.0001_step200_250',
    # 'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step8_11',
    # 'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step8_50',
    # 'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs4',
    # 'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs8',
    # 'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs32',
    # 'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200',
    # 'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step150_200',
    # 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep10_lr_0.01_step200_250',
    # 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep50_lr0.0001_step200_250',
    # 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step8_11',
    # 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step8_50',
    # 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs4',
    # 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs8',
    # 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs32',
    # 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200',
    # 'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step150_200',

    # ####### HTC (DetectoRS_ResNet) model ###########
    # '',
]


subprocess.call(['tmux', 'kill-session', '-t', 'master'])       # Kill previous session
subprocess.call(['tmux', 'new-session', '-s', 'master', '-d'])  # create a new tmux session named 'master'

# send commands to activate conda environment and navigate to directory
subprocess.call(['tmux', 'send-keys', '-t', 'master', 'conda activate master', 'Enter'])
subprocess.call(['tmux', 'send-keys', '-t', 'master', 'cd /home/abadr/master/mmdetection', 'Enter'])

# create a new window for each Python script and run it
for i, exp in enumerate(experiments):
    # create a new window
    # subprocess.call(['tmux', 'new-window', '-t', f'master:{i}'])

    # send commands to activate conda environment, navigate to directory, and run the Python script
    subprocess.call(['tmux', 'send-keys', '-t', f'master', 'conda activate master', 'Enter'])

    # Train Experiment
    subprocess.call(['tmux', 'send-keys', '-t', f'master', 'cd /home/abadr/master/mmdetection', 'Enter'])
    subprocess.call(['tmux', 'send-keys', '-t', f'master', f'python tools/train.py configs/amr/{exp}.py', 'Enter'])

    # Plot training
    # latest_file = subprocess.check_output(f'ls -t mmdetection/benchmark/{exp}/*.log | head -1', shell=True)
    # latest_file = latest_file.rstrip().decode()
    # subprocess.call(['python', 'mmdetection/tools/analysis_tools/analyze_logs.py', 'plot_curve',
    #                 f'{latest_file}.json', '--keys', 'mAP', '--legend', 'mAP', '--title', exp,
    #                 '--out', f'{latest_file[:-4]}.png'
    #                 ])
    subprocess.call(['tmux', 'send-keys', '-t', f'master', f'plot {exp}', 'Enter'])

# attach to the tmux session to see the output
subprocess.call(['tmux', 'attach-session', '-t', 'master'])
