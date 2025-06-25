import os
import shutil
import glob
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split  # Missing import

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import timm  # pip install timm

# Define directories
SPLIT_OUTPUT_DIR = "C:\Users\saima\Desktop\code\DEEPFAKE"

REAL_TRAIN_DIR = os.path.join(SPLIT_OUTPUT_DIR, "REAL/train")
REAL_VAL_DIR = os.path.join(SPLIT_OUTPUT_DIR, "REAL/val")
REAL_TEST_DIR = os.path.join(SPLIT_OUTPUT_DIR, "REAL/test")

FAKE_TRAIN_DIR = os.path.join(SPLIT_OUTPUT_DIR, "Fake/train")
FAKE_VAL_DIR = os.path.join(SPLIT_OUTPUT_DIR, "Fake/val")
FAKE_TEST_DIR = os.path.join(SPLIT_OUTPUT_DIR, "Fake/test")

def split_and_balance_videos(source_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.1, balance_ratio=None):
    """
    Splits videos from the source into train, validation, and test sets. Optionally balances the data.
    
    Args:
    source_dir (str): Source directory containing videos.
    train_dir (str): Directory to store training videos.
    val_dir (str): Directory to store validation videos.
    test_dir (str): Directory to store test videos.
    train_ratio (float): Proportion of videos for training.
    val_ratio (float): Proportion of videos for validation.
    balance_ratio (float): Proportion of videos to keep for balancing (only for Fake videos).
    
    Returns:
    None
    """
    # Read videos from the source directory
    videos = [os.path.join(source_dir, video) for video in os.listdir(source_dir) if video.endswith(('.mp4', '.avi', '.mov'))]
    
    if not videos:
        print(f"No videos found in {source_dir}")
        return

    # Apply balance if balance_ratio is set (for Fake videos)
    if balance_ratio is not None:
        videos = train_test_split(videos, test_size=balance_ratio, random_state=42)[1]

    # Split videos into train and remaining sets
    train_videos, remaining_videos = train_test_split(videos, test_size=(1 - train_ratio), random_state=42)
    
    # Split remaining videos into validation and test sets
    val_ratio_adjusted = val_ratio / (1 - train_ratio)  # Adjusted validation ratio
    val_videos, test_videos = train_test_split(remaining_videos, test_size=(1 - val_ratio_adjusted), random_state=42)

    # Copy videos to respective directories
    for video_set, output_dir in zip([train_videos, val_videos, test_videos], [train_dir, val_dir, test_dir]):
        os.makedirs(output_dir, exist_ok=True)
        for video_path in video_set:
            shutil.copy(video_path, output_dir)

    print(f"Splitting completed for {source_dir}:")
    print(f"  Train: {len(train_videos)} videos -> {train_dir}")
    print(f"  Validation: {len(val_videos)} videos -> {val_dir}")
    print(f"  Test: {len(test_videos)} videos -> {test_dir}")

# Define video paths
real_videos_path = "path/to/real_videos"
fake_videos_path = "path/to/fake_videos"

# Run the splitting function
split_and_balance_videos(real_videos_path, REAL_TRAIN_DIR, REAL_VAL_DIR, REAL_TEST_DIR)
split_and_balance_videos(fake_videos_path, FAKE_TRAIN_DIR, FAKE_VAL_DIR, FAKE_TEST_DIR, balance_ratio=0.11)


########################################
# 1. FRAME EXTRACTION FUNCTIONS
########################################

def extract_frames(video_path, output_dir, frame_skip=30):
    """
    Extract frames from a video every `frame_skip` frames and save to output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            # Save frame as JPEG. The filename includes the original video name and frame count.
            frame_filename = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}_frame{frame_count}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")


def process_video_folder(video_folder, output_folder, frame_skip=30):
    """
    Process all MP4 videos in a folder and extract frames.
    """
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    for video_file in video_files:
        extract_frames(video_file, output_folder, frame_skip=frame_skip)


def create_csv_from_images(image_folder, label, csv_filename):
    """
    Create a CSV file mapping all JPEG images in a folder to a given label.
    """
    image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))
    data = {"image_path": image_paths, "label": [label] * len(image_paths)}
    df = pd.DataFrame(data)
    df.to_csv(csv_filename, index=False)
    print(f"CSV saved to {csv_filename} with {len(image_paths)} entries.")


########################################
# 2. CUSTOM DATASET CLASS
########################################

class DeepfakeImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        label = int(self.data.iloc[idx]['label'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


########################################
# 3. TRAINING FUNCTION
########################################

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct_preds.double() / total
        print(f"Epoch {epoch+1}/{num_epochs} -> Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return model


########################################
# 4. TESTING (VIDEO PREDICTION) FUNCTION
########################################

def predict_video(video_path, model, transform, device, frame_skip=30):
    """
    Process a video, extract frames, run inference on each frame,
    and return the majority vote prediction.
    """
    model.eval()
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    predictions = []
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                # Convert BGR (OpenCV) to RGB and create a PIL image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                input_tensor = transform(pil_image).unsqueeze(0).to(device)
                outputs = model(input_tensor)
                _, pred = torch.max(outputs, 1)
                predictions.append(pred.item())
            frame_count += 1
    cap.release()
    
    if len(predictions) == 0:
        return None
    # Use majority vote among frame predictions
    prediction = max(set(predictions), key=predictions.count)
    return prediction


########################################
# 5. MAIN FUNCTION
########################################

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection with Vision Transformer (ViT Base)")
    parser.add_argument("--mode", type=str, required=True, choices=["extract", "train", "test"],
                        help="Mode: 'extract' to extract frames, 'train' to train model, 'test' to test a video")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root folder containing video subfolders: Celeb-real, Celeb-synthesis, YouTube-real")
    parser.add_argument("--output_frames", type=str, default="./frames",
                        help="Folder where extracted frames will be saved")
    parser.add_argument("--csv_file", type=str, default="dataset.csv",
                        help="CSV file that will map image paths to labels")
    parser.add_argument("--test_video", type=str, help="Path to a test video (for mode 'test')")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == "extract":
        # Define video folders based on the dataset structure
        celeb_real_folder = os.path.join(args.data_root, "Celeb-real")
        celeb_synthesis_folder = os.path.join(args.data_root, "Celeb-synthesis")
        youtube_real_folder = os.path.join(args.data_root, "YouTube-real")
        
        # Create output subfolders for extracted frames
        real_frames_folder = os.path.join(args.output_frames, "real")
        deepfake_frames_folder = os.path.join(args.output_frames, "deepfake")
        os.makedirs(real_frames_folder, exist_ok=True)
        os.makedirs(deepfake_frames_folder, exist_ok=True)
        
        print("Extracting frames from real videos...")
        process_video_folder(celeb_real_folder, real_frames_folder, frame_skip=30)
        process_video_folder(youtube_real_folder, real_frames_folder, frame_skip=30)
        
        print("Extracting frames from deepfake videos...")
        process_video_folder(celeb_synthesis_folder, deepfake_frames_folder, frame_skip=30)
        
        # Create CSV files for both classes
        create_csv_from_images(real_frames_folder, label=0, csv_filename="real.csv")
        create_csv_from_images(deepfake_frames_folder, label=1, csv_filename="deepfake.csv")
        
        # Merge the two CSVs into one combined dataset
        df_real = pd.read_csv("real.csv")
        df_deepfake = pd.read_csv("deepfake.csv")
        df = pd.concat([df_real, df_deepfake], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows
        df.to_csv(args.csv_file, index=False)
        print(f"Combined dataset CSV saved as {args.csv_file}")
    
    elif args.mode == "train":
        # Define training transforms
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Normalization parameters used in many pretrained models (e.g., ImageNet)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # Create the dataset and dataloader
        dataset = DeepfakeImageDataset(args.csv_file, transform=transform_train)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        
        # Load the pretrained ViT base model and modify the classification head for 2 classes.
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        print("Starting training...")
        model = train_model(model, dataloader, criterion, optimizer, device, num_epochs=args.num_epochs)
        
        # Save the trained model
        torch.save(model.state_dict(), "vit_deepfake_detector.pth")
        print("Training complete. Model saved as vit_deepfake_detector.pth")
    
    elif args.mode == "test":
        if args.test_video is None:
            print("Please provide --test_video <path> for testing mode.")
            return
        
        # Load the model (make sure the same architecture is used)
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
        model.load_state_dict(torch.load("vit_deepfake_detector.pth", map_location=device))
        model.to(device)
        
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        prediction = predict_video(args.test_video, model, transform_test, device, frame_skip=30)
        if prediction is None:
            print("No frames were extracted from the video!")
        else:
            # Label mapping: 0 = Real, 1 = Deepfake
            label_str = "Real" if prediction == 0 else "Deepfake"
            print(f"Predicted label for video '{args.test_video}': {label_str}")


if __name__ == "__main__":
    main()
