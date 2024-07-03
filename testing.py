import os
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from tqdm import tqdm
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('jiviai/RSNA_Refined')

output_dir = '/home/jayant/Desktop/jivi/sig_lip_train/training_images'
os.makedirs(output_dir, exist_ok=True)

def process_image(i, c):
    image = i['image']
    metadata = PngInfo()
    metadata.add_text("caption", i["caption"])
    image_path = os.path.join(output_dir, f"{c}.png")
    image.save(image_path, pnginfo=metadata)

# Use ThreadPoolExecutor to parallelize the image processing
with ThreadPoolExecutor(max_workers=18) as executor:
    futures = []
    for c, i in enumerate(tqdm(dataset['train'], total=len(dataset['train']))):
        futures.append(executor.submit(process_image, i, c))

    # Wait for all futures to complete
    for future in tqdm(futures, desc="Processing images"):
        future.result()

print("All images processed and saved successfully.")
