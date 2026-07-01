import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import subprocess

def prepare_data():
    dataset_dir = "/kaggle/input/ucf101/UCF101/UCF-101"
    
    print(f"Checking for dataset at {dataset_dir}...")
    if not os.path.exists(dataset_dir):
        print(f"Warning: {dataset_dir} not found. Please make sure the UCF101 dataset is mounted.")
        return False
        
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    data = []
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        for video_file in glob.glob(os.path.join(cls_dir, "*.avi")):
            data.append((video_file, class_to_idx[cls]))
            
    if len(data) == 0:
        print("No .avi files found. Please check dataset path.")
        return False

    df = pd.DataFrame(data, columns=["path", "label"])
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    
    train_df.to_csv("/kaggle/working/ucf101_train_paths.csv", index=False, header=False, sep=" ")
    val_df.to_csv("/kaggle/working/ucf101_val_paths.csv", index=False, header=False, sep=" ")
    print(f"Created CSVs successfully: {len(train_df)} training, {len(val_df)} validation videos.")
    return True

def run_training():
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Run pre-training (V-JEPA) on 2 GPUs
    cmd = [
        "python", "app/main.py",
        "--fname", "configs/train/vitt16/ucf101-pretrain.yaml",
        "--devices", "cuda:0", "cuda:1"
    ]
    
    print(f"\\n🚀 Kicking off training with command: {' '.join(cmd)}\\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    if prepare_data():
        run_training()
    else:
        print("Data preparation failed. Training aborted.")
