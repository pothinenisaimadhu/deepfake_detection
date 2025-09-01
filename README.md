# Deepfake Detection using XceptionNet

This repository provides a PyTorch implementation for **deepfake image classification** using the **XceptionNet** architecture (via `timm`). The project is built on the **Celeb-DF v2 dataset**, which contains both real and synthetically generated (fake) celebrity face images.  

The goal is to train a binary classifier to distinguish between **real** and **fake** images.

---

## 🚀 Features
- Custom **PyTorch Dataset** class to handle Celeb-DF images.  
- Image preprocessing and augmentation with **Torchvision Transforms**.  
- Train/validation split with `DataLoader` for efficient batching.  
- Model definition using **XceptionNet** (pretrained on ImageNet via `timm`).  
- Binary classification with **BCEWithLogitsLoss**.  
- Accuracy and loss tracking for both training and validation sets.  
- Visualization of training curves (loss & accuracy).  
- Model saving for later inference or fine-tuning.

---

## 📂 Dataset
The project uses the **Celeb-DF v2** dataset, with images extracted from the video frames.  
Directory structure should look like this:

C:/792/Celeb-DF-v2_images/
│── Celeb-real_images/
│ ├── img1.jpg
│ ├── img2.jpg
│ └── ...
│
└── Celeb-synthesis_images/
├── fake1.jpg
├── fake2.jpg
└── ...

yaml
Copy code

- `Celeb-real_images` → contains real celebrity images.  
- `Celeb-synthesis_images` → contains fake (deepfake) images.  

---

## ⚙️ Requirements
Install the following Python packages:

```bash
pip install torch torchvision timm opencv-python matplotlib pandas numpy pillow
🛠️ Training Workflow
Dataset Loading

Uses CustomDataset to load images and map them to labels:

0 → Real

1 → Fake

Preprocessing

Resize images to 128x128.

Convert to tensors.

Data Split

80% training

20% validation

Model

XceptionNet (timm.create_model('xception', pretrained=True, num_classes=1)).

Training Loop

Optimizer: Adam with learning rate 1e-4.

Loss: BCEWithLogitsLoss.

Tracks accuracy and loss across epochs.

Visualization

Plot training/validation accuracy and loss vs. epochs.

Model Saving

Saves model state, optimizer state, and training history in CelebDF.pt.

📊 Example Outputs
Training & validation accuracy curves.

Loss vs. epoch curves.

Final saved model (CelebDF.pt).

🐞 Common Issues
num_samples=0 error → occurs when the dataset path is empty or not structured properly.
✅ Ensure C:/792/Celeb-DF-v2_images/ contains Celeb-real_images/ and Celeb-synthesis_images/ folders with .jpg images.

▶️ Usage
Clone the repo and place dataset in the correct directory.

Run the training script (e.g., in Jupyter/Colab).

Plot training results with:

python
Copy code
plot_curves(train_losses, train_accs, val_losses, val_accs)
Use the saved model for inference or fine-tuning.

🔮 Inference: Predicting Real vs Fake
🔹 Single Image Inference
python
Copy code
import torch
from torchvision import transforms
from PIL import Image
import timm

# ===========================
# 1. Load Model Checkpoint
# ===========================
checkpoint = torch.load("CelebDF.pt", map_location="cuda")

# Recreate the model architecture
model = timm.create_model("xception", pretrained=False, num_classes=1)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval().to("cuda")

# ===========================
# 2. Define Transform
# ===========================
inference_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ===========================
# 3. Load and Preprocess Image
# ===========================
img_path = "sample.jpg"   # replace with your test image
img = Image.open(img_path).convert("RGB")
img_tensor = inference_transform(img).unsqueeze(0).to("cuda")

# ===========================
# 4. Make Prediction
# ===========================
with torch.no_grad():
    output = model(img_tensor.float())
    prob = torch.sigmoid(output).item()

prediction = "Fake" if prob > 0.5 else "Real"

print(f"Prediction: {prediction} (Confidence: {prob:.4f})")
✅ Example Output:

makefile
Copy code
Prediction: Fake (Confidence: 0.8723)
🔹 Batch Inference (Folder of Images)
python
Copy code
import os

folder_path = "test_images/"   # folder with .jpg images

for file in os.listdir(folder_path):
    if file.endswith(".jpg"):
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path).convert("RGB")
        img_tensor = inference_transform(img).unsqueeze(0).to("cuda")

        with torch.no_grad():
            output = model(img_tensor.float())
            prob = torch.sigmoid(output).item()

        prediction = "Fake" if prob > 0.5 else "Real"
        print(f"{file} → {prediction} (Confidence: {prob:.4f})")
✅ Example Output:

less
Copy code
img1.jpg → Real (Confidence: 0.1234)
img2.jpg → Fake (Confidence: 0.8912)
img3.jpg → Fake (Confidence: 0.7654)
📌 Next Steps
Add data augmentation for robustness.

Implement test-time evaluation.

Explore lightweight models for real-time inference.

Extend to video-based detection.

📜 License
This project is for research and educational purposes only.

yaml
Copy code

---

👉 Do you also want me to include a **pretrained weights download link** in the README (so users don’t always need to train from scratch)?
