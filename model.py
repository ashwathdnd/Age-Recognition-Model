# model.py
import torch.nn as nn

class AgeNet(nn.Module):
    def __init__(self):
        super(AgeNet, self).__init__()
        
        # Define age ranges
        self.age_ranges = [
            (0, 2),    # Babies
            (3, 6),    # Toddlers
            (7, 12),   # Children
            (13, 19),  # Teenagers
            (20, 29),  # Young Adults
            (30, 39),  # Thirties
            (40, 49),  # Forties
            (50, 59),  # Fifties
            (60, 69),  # Sixties
            (70, 100)  # Elderly
        ]
        num_classes = len(self.age_ranges)
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)  # Output probabilities for each age range
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x