import os
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm.auto import tqdm
from train_with_head_distributed import Classifier, load_split_train_test, CONFIG

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model = Classifier(num_classes=CONFIG['num_classes'])
model = torch.nn.DataParallel(model)  # Use DataParallel to distribute across GPUs
model = model.to(device)

# Load the saved model weights
weights_path = 'path_to_weights/best.pth'  # Update this with your weights path
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()  # Set the model to evaluation mode

# Load test data
_, testloader = load_split_train_test(CONFIG['DATA_DIR'])

def evaluate_model():
    """
    Evaluates the model on the test dataset and computes accuracy and confusion matrix.
    """
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Accumulate correct predictions
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Collect predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

    # Generate and display the confusion matrix
    plot_confusion_matrix(all_labels, all_preds, classes=list(range(CONFIG['num_classes'])))
    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(CONFIG['num_classes'])]))

def plot_confusion_matrix(true_labels, pred_labels, classes):
    """
    Plots the confusion matrix.
    """
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
