import os
import cv2
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm

class DeepfakeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def extract_frames(video_path, output_dir, frame_skip=30):
    """Extract frames from a video file"""
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
            frame_path = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}_frame{saved_count}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        frame_count += 1
        
    cap.release()
    return saved_count

def prepare_dataset(base_dir, frames_output_dir):
    """Prepare dataset from the given directory structure"""
    # Create output directories
    os.makedirs(frames_output_dir, exist_ok=True)
    real_frames_dir = os.path.join(frames_output_dir, 'real')
    fake_frames_dir = os.path.join(frames_output_dir, 'fake')
    os.makedirs(real_frames_dir, exist_ok=True)
    os.makedirs(fake_frames_dir, exist_ok=True)

    # Process real videos
    celeb_real_dir = os.path.join(base_dir, r'C:\Users\saima\Desktop\code\DEEPFAKE\Celeb-real')
    youtube_real_dir = os.path.join(base_dir, r'C:\Users\saima\Desktop\code\DEEPFAKE\YouTube-real')
    fake_dir = os.path.join(base_dir, r'C:\Users\saima\Desktop\code\DEEPFAKE\Celeb-synthesis')

    image_paths = []
    labels = []

    # Process real videos from Celeb-real
    for video in os.listdir(celeb_real_dir):
        if video.endswith('.mp4'):
            video_path = os.path.join(celeb_real_dir, video)
            frames = extract_frames(video_path, real_frames_dir)
            for i in range(frames):
                frame_path = os.path.join(real_frames_dir, f"{os.path.splitext(video)[0]}_frame{i}.jpg")
                image_paths.append(frame_path)
                labels.append(0)  # 0 for real

    # Process real videos from YouTube-real
    for video in os.listdir(youtube_real_dir):
        if video.endswith('.mp4'):
            video_path = os.path.join(youtube_real_dir, video)
            frames = extract_frames(video_path, real_frames_dir)
            for i in range(frames):
                frame_path = os.path.join(real_frames_dir, f"{os.path.splitext(video)[0]}_frame{i}.jpg")
                image_paths.append(frame_path)
                labels.append(0)  # 0 for real

    # Process fake videos
    for video in os.listdir(fake_dir):
        if video.endswith('.mp4'):
            video_path = os.path.join(fake_dir, video)
            frames = extract_frames(video_path, fake_frames_dir)
            for i in range(frames):
                frame_path = os.path.join(fake_frames_dir, f"{os.path.splitext(video)[0]}_frame{i}.jpg")
                image_paths.append(frame_path)
                labels.append(1)  # 1 for fake

    return image_paths, labels

def test_video(model, video_path, transform, device):
    """Test a single video for deepfake detection"""
    model.eval()
    cap = cv2.VideoCapture(video_path)
    predictions = []
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Preprocess and predict
            frame_tensor = transform(frame_pil).unsqueeze(0).to(device)
            output = model(frame_tensor)
            pred = torch.argmax(output, dim=1).item()
            predictions.append(pred)
    
    cap.release()
    
    # Return majority vote
    if len(predictions) == 0:
        return None
    return max(set(predictions), key=predictions.count)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set paths
    base_dir = "./data"  # Directory containing Celeb-real, Celeb-synthesis, YouTube-real folders
    frames_dir = "./frames"  # Directory to store extracted frames
    test_list_path = "List_of_testing_videos.txt"  # Path to test video list

    # Prepare dataset
    print("Preparing dataset...")
    image_paths, labels = prepare_dataset(base_dir, frames_dir)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = DeepfakeDataset(image_paths, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Initialize model
    print("Initializing ViT model...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
    model = model.to(device)

    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 10

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(dataloader):.4f} Accuracy: {accuracy:.2f}%')

    # Save the model
    print("Saving model...")
    torch.save(model.state_dict(), 'deepfake_detector.pth')

    # Test on videos from test list
    print("Testing videos...")
    with open(test_list_path, 'r') as f:
        test_videos = f.read().splitlines()

    for video_path in test_videos:
        prediction = test_video(model, video_path, transform, device)
        if prediction is not None:
            result = "FAKE" if prediction == 1 else "REAL"
            print(f"Video {video_path}: {result}")

if __name__ == "__main__":
    main()