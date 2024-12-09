# find the best model
import os
import json
import copy
# Add these imports at the top
import matplotlib.pyplot as plt
import numpy as np

# Text type performance metrics:
# Driving area Segment: Acc(0.974)    IOU (0.860)    mIOU(0.914)
# Lane line Segment: Acc(0.701)    IOU (0.269)  mIOU(0.626)
# Detect: P(0.113)  R(0.879)  mAP@0.5(0.757)  mAP@0.5:0.95(0.440)
# Time: inference(0.0019s/frame)  nms(0.0002s/frame)
# 2024-12-09 07:11:17,340 => saving checkpoint to runs/BddDataset/_2024-12-06-22-32/epoch-240.pth

# JSON type performance metrics
# [
#   {
#     "Driving area Segment": {
#         "Acc": 0.974,
#         "IOU": 0.860,
#         "mIOU": 0.914
#     },
#     "Lane line Segment": {    
#         "Acc": 0.701,
#         "IOU": 0.269,
#         "mIOU": 0.626
#     },
#     "Detect": {
#         "P": 0.113,
#         "R": 0.879,
#         "mAP05": 0.757,
#         "mAP0595": 0.440
#     },
#     "Time": {
#         "inference": 0.0019,
#         "nms": 0.0002
#     },
#     "path": "runs/BddDataset/_2024-12-06-22-32/epoch-240.pth"
#   }
# ]



RUNS_DIR = 'runs/BddDataset/_2024-12-06-22-32'


def filter_performance_matrics(runs_dir, debug=False):
    # find the log file named with a _train.log ending.
    original_log_file = [f for f in os.listdir(runs_dir) if f.endswith('_train.log')][0]
    original_log_file_path = os.path.join(runs_dir, original_log_file)

    os.system(f'grep -E "Driving area Segment|Lane line Segment|Detect|inference|saving checkpoint to" {original_log_file_path} > {runs_dir}/log_filtered.txt')

    # convert the log_filtered.txt to JSON type performance metrics
    performance_metrics = []
    # 1. read the log_filtered.txt
    with open(os.path.join(runs_dir, 'log_filtered.txt'), 'r') as f:
        log_filtered_lines = f.readlines()

        # extract the performance metrics from the log_filtered.txt
        # 1. Driving area Segment
        # 2. Lane line Segment
        # 3. Detect
        # 4. Time
        # 5. path
        metrics = {}


        epoch = 0
        for line in log_filtered_lines:
            if debug:
                print(line)
            # 1. Driving area Segment
            if line.startswith('Driving area Segment'):
                # reset the metrics
                metrics['Driving area Segment'] = {}
                metrics['Lane line Segment'] = {}
                metrics['Detect'] = {}
                metrics['Time'] = {}
                metrics['path'] = ""
                epoch += 1
                # Driving area Segment: Acc(0.974)    IOU (0.860)    mIOU(0.914)
                datas = line.split('(')
                if debug:
                    print(datas)
                datas2 = [float(data.strip().split(')')[0]) for data in datas[1:]]
                if debug:
                    print(datas2)
                metrics['Driving area Segment']['Acc'] = datas2[0]
                metrics['Driving area Segment']['IOU'] = datas2[1]
                metrics['Driving area Segment']['mIOU'] = datas2[2]
                print(metrics)
            elif line.startswith('Lane line Segment'):
                # Lane line Segment: Acc(0.701)    IOU (0.269)  mIOU(0.626)
                datas = line.split('(')
                datas2 = [float(data.strip().split(')')[0]) for data in datas[1:]]
                if debug:
                    print(datas2)
                metrics['Lane line Segment']['Acc'] = datas2[0]
                metrics['Lane line Segment']['IOU'] = datas2[1]
                metrics['Lane line Segment']['mIOU'] = datas2[2]
                if debug:
                    print(metrics)
            elif line.startswith('Detect'):
                # Detect: P(0.113)  R(0.879)  mAP@0.5(0.757)  mAP@0.5:0.95(0.440)
                datas = line.split('(')
                datas2 = [float(data.strip().split(')')[0]) for data in datas[1:]]
                if debug:
                    print(datas2)
                metrics['Detect']['P'] = datas2[0]
                metrics['Detect']['R'] = datas2[1]
                metrics['Detect']['mAP05'] = datas2[2]
                metrics['Detect']['mAP0595'] = datas2[3]
                if debug:
                    print(metrics)
            elif line.startswith('Time: inference'):
                # Time: inference(0.0019s/frame)  nms(0.0002s/frame)
                datas = line.split('(')
                datas2 = [float(data.strip().split('s')[0]) for data in datas[1:]]
                if debug:
                    print(datas2)
                metrics['Time']['inference'] = datas2[0]
                metrics['Time']['nms'] = datas2[1]
                if debug:
                    print(metrics)
            elif line.endswith('.pth\n'):
                if debug:
                    print(line)
                #2024-12-09 07:11:17,340 => saving checkpoint to runs/BddDataset/_2024-12-06-22-32/epoch-240.pth
                datas = line.split(' ')
                if debug:
                    print(datas)
                metrics['path'] = datas[-1].strip()
                if debug:
                    print(metrics)
                # check if the epoch number is matched
                if not f"epoch-{epoch}.pth" in metrics['path']:
                    print(f"Error epoch {epoch} is not matched!!!!!")
                    continue
                # should use deepcopy to avoid the reference problem
                performance_metrics.append(copy.deepcopy(metrics))

        if debug:
            print(json.dumps(performance_metrics, indent=4))
        
        return performance_metrics
        
    

def draw_performance_metrics(performance_metrics):
    # Extract epochs and metrics
    epochs = range(1, len(performance_metrics) + 1)
    
    # Create metrics arrays
    driving_acc = [m['Driving area Segment']['Acc'] for m in performance_metrics]
    lane_acc = [m['Lane line Segment']['Acc'] for m in performance_metrics]
    detect_map = [m['Detect']['mAP05'] for m in performance_metrics]
    
    # Create figure with multiple subplots
    plt.figure(figsize=(15, 10))
    
    # Plot metrics
    plt.subplot(2, 2, 1)
    plt.plot(epochs, driving_acc, 'b-', label='Driving Area Accuracy')
    plt.title('Driving Area Segmentation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, lane_acc, 'g-', label='Lane Line Accuracy')
    plt.title('Lane Line Segmentation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, detect_map, 'r-', label='Detection mAP@0.5')
    plt.title('Detection mAP@0.5')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.grid(True)
    plt.legend()
    
    # Add a text box with best metrics
    best_epoch = np.argmax(detect_map) + 1  # +1 because epochs start from 1
    # show all the metrics in the best epoch
    best_metrics = (
        f"Best Metrics at Epoch {best_epoch} according to detection mAP@0.5:\n"
        f"{json.dumps(performance_metrics[best_epoch-1], indent=4)}\n"
    )
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.5, best_metrics, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, 'performance_metrics.png'))
    plt.close()

    return best_epoch   


if __name__ == '__main__':
    performance_metrics = filter_performance_matrics(RUNS_DIR, debug=False)
    print(json.dumps(performance_metrics, indent=4))
    best_epoch = draw_performance_metrics(performance_metrics)
    print(f"The best epoch is {best_epoch}")
    #save the best model
    best_model_path = performance_metrics[best_epoch-1]['path']
    os.system(f'cp {best_model_path} {RUNS_DIR}/best_model.pth')
    print(f"The best model is saved to {RUNS_DIR}/best_model.pth")