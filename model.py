
import torch
import torch.nn as nn
from dataset import *
from torchvision import models, transforms
from PIL import Image
from torch.utils.data import  DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools


# Early stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_score is None or self.best_score - val_loss > self.min_delta:
            self.best_score = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Training function with hyperparameter tuning

def train_model(data_dir, epochs=30, batch_size=16, lr_list=[1e-4], hidden_dims=[(512, 256)]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = MultiLabelClothingDataset(data_dir, transform=transform)
    indices = list(range(len(dataset)))
    if len(indices) == 0:
        raise ValueError("No data found in dataset.")
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    label_freq = dataset.label_counts / len(dataset)
    pos_weights = 1.0 / (label_freq + 1e-6)
    pos_weights = pos_weights.to(device)

    best_f1 = 0
    best_setting = None

    for lr, (h1, h2) in itertools.product(lr_list, hidden_dims):
        print(f"\nTrying config: LR={lr}, Hidden dims=({h1}, {h2})")

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, len(ALL_TAGS))
        )
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        early_stopper = EarlyStopping(patience=5, min_delta=0.001)

        loss_list = []
        f1_list = []

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - LR={lr}"):
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    logits = model(imgs)
                    outputs = torch.sigmoid(logits)
                    preds = (outputs >= 0.5).float()
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            epoch_loss = running_loss / len(train_loader)
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            macro_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            macro_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

            loss_list.append(epoch_loss)
            f1_list.append(macro_f1)

            print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f} | Macro F1 = {macro_f1:.4f} | Precision = {macro_prec:.4f} | Recall = {macro_recall:.4f}")

            print("Per-label F1, Precision, Recall:")
            per_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
            per_prec = precision_score(all_labels, all_preds, average=None, zero_division=0)
            per_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
            for i, tag in enumerate(ALL_TAGS):
                print(f"  {tag:20s} | F1: {per_f1[i]:.2f} | Prec: {per_prec[i]:.2f} | Recall: {per_recall[i]:.2f}")

            early_stopper(epoch_loss, model)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_setting = (lr, h1, h2)
            torch.save(model.state_dict(), "structure_classifier_best.pth")

        plt.figure()
        plt.plot(range(1, len(loss_list)+1), loss_list, label='Loss')
        plt.plot(range(1, len(f1_list)+1), f1_list, label='Macro F1')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title(f'LR={lr}, Hidden=({h1},{h2})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"training_metrics_lr{lr}_h{h1}_{h2}.png")

    print(f"\nBest config: LR={best_setting[0]}, Hidden=({best_setting[1]}, {best_setting[2]}) | Macro F1 = {best_f1:.4f}")
    return best_setting

# Single image inference
def predict_single_image(image_path, model_path="structure_classifier_best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, len(ALL_TAGS))
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().squeeze().numpy()

    results = [(tag, float(prob)) for tag, prob in zip(ALL_TAGS, probs) if prob >= 0.5]
    print("\nPredicted tags:")
    for tag, prob in results:
        print(f"  {tag:20s}  |  confidence = {prob:.2f}")

    return results
