import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class AgeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.age_ranges = [
            (0, 2), (3, 6), (7, 12), (13, 19), (20, 29),
            (30, 39), (40, 49), (50, 59), (60, 69), (70, 100)
        ]
    
    def get_age_range_index(self, age):
        for i, (min_age, max_age) in enumerate(self.age_ranges):
            if min_age <= age <= max_age:
                return i
        return len(self.age_ranges) - 1
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        # Get age from filename
        age = int(self.image_files[idx].split('_')[0])
        # Convert to age range index
        age_range_idx = self.get_age_range_index(age)
        
        if self.transform:
            image = self.transform(image)
        
        return image, age_range_idx