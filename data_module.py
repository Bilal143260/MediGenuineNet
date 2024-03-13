import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip,RandomVerticalFlip, RandomRotation, RandomResizedCrop, ColorJitter, RandomAffine
from PIL import Image 

transform = Compose([
    ToTensor(),  # Convert images to Tensor
    Normalize([0.5], [0.5]),  # Assuming single-channel (grayscale) images; adjust for multi-channel
    RandomHorizontalFlip(p=0.2),
    RandomVerticalFlip(p=0.2),
    RandomRotation(degrees=15),  # Rotate +/- 15 degrees
    RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)), 
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10)  # Apply affine transformation
])

class BinaryImageDataset(Dataset):
    def __init__(self, synthetic_dir, original_dir, transform=transform):
        """
        Args:
            synthetic_dir (string): Directory with all the images for label 0.
            original_dir (string): Directory with all the images for label 1.
        """
        self.synthetic_dir = synthetic_dir
        self.original_dir = original_dir
        self.label0_images = [os.path.join(synthetic_dir, file) for file in os.listdir(synthetic_dir)]
        self.label1_images = [os.path.join(original_dir, file) for file in os.listdir(original_dir)]
        self.total_images = self.label0_images + self.label1_images
        self.labels = [0] * len(self.label0_images) + [1] * len(self.label1_images)
        
        self.transform = transform

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        image_path = self.total_images[idx]
        image = Image.open(image_path)
        label = self.labels[idx]
        image = self.transform(image)
        return image, label

    
train_synthetic_data_path = 'dataset/train/synthetic'
train_orginal_data_path = 'dataset/train/original'
train_dataset = BinaryImageDataset(train_synthetic_data_path, train_orginal_data_path)

test_synthetic_data_path = 'dataset/test/synthetic'
test_orginal_data_path = 'dataset/test/original'
test_dataset = BinaryImageDataset(test_synthetic_data_path, test_orginal_data_path)


