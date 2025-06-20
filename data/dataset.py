# Dataset loading and handling logic 
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class AgeDataset(Dataset):
    """Custom dataset for age classification"""
    def __init__(self, root_dir, transform=None, regression=True):
        self.root_dir = root_dir
        self.transform = transform
        self.regression = regression
        self.images = []
        self.labels = []
        age_folders = ['20', '30', '40', '50', '60', '70', '80']
        for age_folder in age_folders:
            folder_path = os.path.join(root_dir, age_folder)
            if os.path.exists(folder_path):
                age = int(age_folder)
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(folder_path, img_file)
                        self.images.append(img_path)
                        if regression:
                            self.labels.append(float(age))
                        else:
                            self.labels.append(age_folders.index(age_folder))
        print(f"Loaded {len(self.images)} images")
        print(f"Age distribution: {dict(zip(*np.unique(self.labels, return_counts=True)))}")
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32 if self.regression else torch.long)

class TTA_Dataset(Dataset):
    """Test Time Augmentation Dataset Wrapper"""
    def __init__(self, base_dataset, n_augmentations=5):
        self.base_dataset = base_dataset
        self.n_augmentations = n_augmentations
        self.tta_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.95, 1.0)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.base_dataset) * self.n_augmentations
    def __getitem__(self, idx):
        base_idx = idx // self.n_augmentations
        img_path = self.base_dataset.images[base_idx]
        label = self.base_dataset.labels[base_idx]
        image = Image.open(img_path).convert('RGB')
        image = self.tta_transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

def create_augmented_dataset(dataset, multiplier=3):
    """Create additional samples through augmentation for training"""
    augmented_images = []
    augmented_labels = []
    strong_transform = None  # Not used directly, transform is applied in DataLoader
    print(f"Creating {multiplier}x augmented dataset...")
    for idx in range(len(dataset)):
        img_path = dataset.images[idx]
        label = dataset.labels[idx]
        augmented_images.append(img_path)
        augmented_labels.append(label)
        for _ in range(multiplier - 1):
            augmented_images.append(img_path)
            augmented_labels.append(label)
    dataset.images = augmented_images
    dataset.labels = augmented_labels
    print(f"Dataset size increased from {len(dataset)//multiplier} to {len(dataset)}")
    return dataset 