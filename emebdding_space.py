import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image

# Load the embeddings from the JSON file
with open('/home/jayant/Desktop/jivi/sig_lip_train/embedding_space.json', 'r') as f:
    data = json.load(f)

# Extract the embeddings and filenames
embeddings_healthy = []
embeddings_unhealthy = []
for item in data:
    for name, embedding in item.items():
        if isinstance(embedding, list):
            # Flatten the embedding if needed and convert to NumPy array
            output_dir = '/home/jayant/Desktop/jivi/sig_lip_train/training_images'
            image_path = os.path.join(output_dir, name)
            img = Image.open(image_path)
            metadata = img.info
            embedding = np.array(embedding).flatten()
            caption = metadata['caption']
            
            if 'healthy' in caption.lower() or 'normal' in caption.lower():
                embeddings_healthy.append(embedding)
            else:
                embeddings_unhealthy.append(embedding)

# Convert embeddings list to a NumPy array
embeddings_healthy = np.array(embeddings_healthy)
embeddings_unhealthy = np.array(embeddings_unhealthy)

# Apply t-SNE to reduce the dimensionality to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=10)
embeddings_combined = np.concatenate((embeddings_healthy, embeddings_unhealthy), axis=0)
embeddings_2d = tsne.fit_transform(embeddings_combined)

# Separate embeddings for healthy and unhealthy
embeddings_2d_0 = embeddings_2d[:len(embeddings_healthy)]
embeddings_2d_1 = embeddings_2d[len(embeddings_healthy):]

# Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d_0[:, 0], embeddings_2d_0[:, 1], alpha=0.5, color='green',label='Healthy')
plt.scatter(embeddings_2d_1[:, 0], embeddings_2d_1[:, 1], alpha=0.5, color='red', label='Unhealthy')
plt.title('t-SNE Visualization of Image Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()
