# Define label groups
import torch
import os

from PIL import Image
from torch.utils.data import Dataset

COLLARS = ["deep_round_neck", "v_neck", "round_neck", "shirt_collar", "turtleneck", "hoodie_hood"]
SLEEVES = ["sleeveless", "short_sleeve", "puffy_short_sleeve", "elbow_sleeve", "long_sleeve", "puffy_long_sleeve"]
HEMS = ["above_belly", "hip_length", "thigh_length", "knee_length", "calf_length"]
ALL_TAGS = COLLARS + SLEEVES + HEMS

# Dataset definition
class MultiLabelClothingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.label_counts = torch.zeros(len(ALL_TAGS))

        for fname in os.listdir(root_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                label_path = os.path.join(root_dir, fname + ".txt")
                if os.path.exists(label_path):
                    with open(label_path, "r") as f:
                        tags = f.read().strip().split(',')
                    label_vec = [1 if tag in tags else 0 for tag in ALL_TAGS]
                    label_tensor = torch.tensor(label_vec, dtype=torch.float)
                    self.label_counts += label_tensor
                    self.samples.append((os.path.join(root_dir, fname), label_tensor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
