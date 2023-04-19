import subprocess

experiments = [

    # ####### YOLOV3 model ###########



    # ####### RetinaNet model ###########
    'balloon_RetinaNet_ResNet50_SGD_lr0.02_mom0.9_wd0.0001_ep1000',
    'leopard_RetinaNet_ResNet50_SGD_lr0.02_mom0.9_wd0.0001_ep1000',
    'leopard_v300_RetinaNet_ResNet50_SGD_lr0.02_mom0.9_wd0.0001_ep1000',

    # ####### HTC model ###########
    'balloon_HybridTaskCascade_ResNet50_SGD_ep10_lr_0.01_step200_250',
    'balloon_HybridTaskCascade_ResNet50_SGD_ep50_lr0.0001_step200_250',
    'balloon_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step8_11',
    'balloon_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs4',
    'balloon_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs8',
    'balloon_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs32',
    'balloon_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200',
    'balloon_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step150_200',
    'leopard_HybridTaskCascade_ResNet50_SGD_ep10_lr_0.01_step200_250',
    'leopard_HybridTaskCascade_ResNet50_SGD_ep50_lr0.0001_step200_250',
    'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step8_11',
    'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs4',
    'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs8',
    'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs32',
    'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200',
    'leopard_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step150_200',
    'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep10_lr_0.01_step200_250',
    'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep50_lr0.0001_step200_250',
    'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step8_11',
    'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs4',
    'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs8',
    'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200_bs32',
    'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step50_200',
    'leopard_v300_HybridTaskCascade_ResNet50_SGD_ep500_lr0.0025_step150_200',

    # ####### HTC (DetectoRS_ResNet) model ###########
    '',
]


subprocess.call(['tmux', 'kill-session', '-t', 'master'])     # Kill previous session
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
    subprocess.call(['tmux', 'send-keys', '-t', f'master', 'cd /home/abadr/master/mmdetection', 'Enter'])
    subprocess.call(['tmux', 'send-keys', '-t', f'master', f'python tools/train.py configs/amr/{exp}.py', 'Enter'])
    subprocess.call(['tmux', 'send-keys', '-t', f'master', f'plot {exp}', 'Enter'])
    subprocess.call(['tmux', 'send-keys', '-t', f'master', f'sleep 5', 'Enter'])

    # wait for the experiment to finish before starting the next one
    # subprocess.call(['tmux', 'wait-for', '-S', f'master:{i}'])

# attach to the tmux session to see the output
subprocess.call(['tmux', 'attach-session', '-t', 'master'])
