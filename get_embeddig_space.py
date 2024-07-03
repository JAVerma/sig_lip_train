import torch
from PIL import Image
import open_clip
from tqdm import tqdm
import glob, os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from concurrent.futures import ThreadPoolExecutor

# Load the model and set it to evaluation mode
model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384')
model.eval()  # Model in train mode by default, impacts some models with BatchNorm or stochastic depth active

# Set device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_image_encoding(img_path):
    """
    Function to encode image into features using the CLIP model.
    """
    image = Image.open(img_path)
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image).cpu().numpy().tolist()
    name = os.path.basename(img_path)
    return image_features, name

# Path to the directory containing images
images_path = glob.glob('/home/jayant/Desktop/jivi/sig_lip_train/training_images/*')

# Check if embeddings already exist in JSON file
existing_embeddings = {}
if os.path.exists('./embedding_space.json'):
    with open('./embedding_space.json', 'r') as json_file:
        existing_data = json.load(json_file)
        for item in existing_data:
            for name, embedding in item.items():
                existing_embeddings[name] = embedding

# Dictionary to hold the image features and names
dic = []

# Iterate over each image path and encode the image
for img_path in tqdm(images_path, desc="Processing images"):
    name = os.path.basename(img_path)
    
    # Skip if already processed
    if name in existing_embeddings:
        print(f"Skipping {name}, already processed.")
        continue
    
    # Get image features and name
    image_features, name = get_image_encoding(img_path)
    
    # Append to dictionary
    dic.append({name: image_features})
    
    # Write to JSON file after each iteration
    with open('./embedding_space.json', 'w') as data_file:
        json.dump(dic, data_file, indent=4)
