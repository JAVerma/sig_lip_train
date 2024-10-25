import os
import torch
import clip
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Directory paths
WEIGHTS_DIR = "/home/jayant/sig_lip_train/weights/young-frog-21-6yqgpayr"
DATA_DIR = "/home/jayant/neurodiscovery/data_mri/mri_image/Test"
num_classes = 2

CONFIG = dict(
    clip_type='ViT-L/14@336px',
    batch_size=256,
    dropout=0.5,
    hid_dim=512,
    activation='relu'
)

# Activation functions
get_activation = {
    'relu': torch.nn.ReLU,
    'elu': torch.nn.ELU,
    'leaky_relu': torch.nn.LeakyReLU
}

# Model Definition
class Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        clip_model, _ = clip.load(CONFIG["clip_type"], device)
        self.clip_model = clip_model.visual.float()

        if hasattr(self.clip_model, 'ln_post'):
            self.clip_model.ln_post = torch.nn.Identity()

        if hasattr(self.clip_model, 'transformer'):
            blocks = list(self.clip_model.transformer.resblocks.children())
            self.clip_model.transformer.resblocks = torch.nn.Sequential(*blocks[:-1])

        # Define the classifier head
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(768, CONFIG['hid_dim']),
            get_activation[CONFIG["activation"]](),
            torch.nn.Dropout(CONFIG["dropout"]),
            torch.nn.Linear(CONFIG['hid_dim'], CONFIG['hid_dim'] // 2),
            get_activation[CONFIG["activation"]](),
            torch.nn.Dropout(CONFIG["dropout"]),
            torch.nn.Linear(CONFIG['hid_dim'] // 2, CONFIG['hid_dim'] // 4),
            get_activation[CONFIG["activation"]](),
            torch.nn.Dropout(CONFIG["dropout"]),
            torch.nn.Linear(CONFIG['hid_dim'] // 4, num_classes)
        ).float()

    def forward(self, x):
        x = x.float()
        x = self.clip_model(x)
        x = x.view(x.size(0), -1)
        return self.cls_head(x)

# Dataset Class
class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.im_paths = glob(os.path.join(root_dir, "*", "*"))
        self.val_transform = A.Compose([
            A.Resize(336, 336),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        self.classes = sorted({os.path.basename(os.path.dirname(p)) for p in self.im_paths})
        self.label_dict = {name: idx for idx, name in enumerate(self.classes)}

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        folder_name = os.path.basename(os.path.dirname(im_path))
        label = self.label_dict[folder_name]

        img = cv2.imread(im_path)
        img = self.val_transform(image=img)['image']
        return img, label, im_path

dataset = ImageDataset(DATA_DIR)
test_loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=8)

# Evaluate Model
def evaluate_model(weights_path):
    """Evaluates the model using the given weights and saves confusion matrix and incorrect predictions."""
    epoch_name = os.path.splitext(os.path.basename(weights_path))[0]  # Extract epoch name

    print(f"\nEvaluating weights: {weights_path}")
    model = Classifier(num_classes)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    correct, total = 0, 0
    all_preds, all_labels = [], []

    incorrect_dir = f'./incorrect_predictions_{epoch_name}'
    os.makedirs(incorrect_dir, exist_ok=True)

    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc=f"Evaluating {epoch_name}"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            save_incorrect_predictions(predicted, labels, paths, incorrect_dir)

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

    cm_filename = f'confusion_matrix_{epoch_name}.png'
    plot_confusion_matrix(all_labels, all_preds, list(range(num_classes)), cm_filename)

    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_classes)]))

def save_incorrect_predictions(predictions, labels, paths, save_dir):
    """Save incorrect predictions with predicted and ground truth labels."""
    for pred, label, path in zip(predictions.cpu().numpy(), labels.cpu().numpy(), paths):
        if pred != label:
            img = cv2.imread(path)
            pred_text = f"Pred: {pred}"
            gt_text = f"GT: {label}"

            cv2.putText(img, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(img, gt_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            save_path = os.path.join(save_dir, os.path.basename(path))
            cv2.imwrite(save_path, img)

def plot_confusion_matrix(true_labels, pred_labels, classes, save_path):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(save_path)  # Save the confusion matrix as an image
    plt.close()

if __name__ == "__main__":
    # Loop through all weight files and evaluate
    weight_files = sorted(glob(os.path.join(WEIGHTS_DIR, "*.pth")))
    for weight_file in weight_files:
        evaluate_model(weight_file)
