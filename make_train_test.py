import glob
import os
import shutil
import random

# Configuration
train_count = 15000  # Number of images to select for training
data_root = '/home/jayant/data_mri/mri_image'
output_root = '/home/jayant/neuro/data_mri/mri_image/Train'

# Create output directories if they don't exist
os.makedirs(f'{output_root}/Train_A', exist_ok=True)
os.makedirs(f'{output_root}/Train_B', exist_ok=True)
os.makedirs(f'{output_root}/Test_A', exist_ok=True)
os.makedirs(f'{output_root}/Test_B', exist_ok=True)

# Gather all images
images = glob.glob(f'{data_root}/*/*')

# Separate images based on folder names
ventricles_images = [img for img in images if 'ventricle_images' in img]
non_ventricles_images = [img for img in images if 'non_ventricles_images' in img]

# Randomly select 15000 images for each class for training
train_A = random.sample(ventricles_images, min(train_count, len(ventricles_images)))
train_B = random.sample(non_ventricles_images, min(train_count, len(non_ventricles_images)))

# Get the remaining images for testing
test_A = list(set(ventricles_images) - set(train_A))
test_B = list(set(non_ventricles_images) - set(train_B))

# Function to copy images to the destination
def copy_images(image_list, destination):
    for img in image_list:
        shutil.copy(img, destination)

# Copy images to their respective directories
print("Copying Train_A images...")
copy_images(train_A, f'{output_root}/Train_A')

print("Copying Train_B images...")
copy_images(train_B, f'{output_root}/Train_B')

print("Copying Test_A images...")
copy_images(test_A, f'{output_root}/Test_A')

print("Copying Test_B images...")
copy_images(test_B, f'{output_root}/Test_B')

print("Data split complete.")
