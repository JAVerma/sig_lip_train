import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset


class multiclassdataset(Dataset):
    # def __init__(self, hfdataset, image_processor, text_Processor, transform=None):
    #     self.augmentations = transform
    #     self.images = load_dataset(hfdataset)["train"]["image"]
    #     self.texts = load_dataset(hfdataset)["train"]["caption"]
    #     self.processor = image_processor
    #     self.text_Processor = text_Processor

    def __init__(self, hfdataset, processor, transform=None):
        self.augmentations = transform
        self.images = load_dataset(hfdataset)["train"]["image"]
        self.texts = load_dataset(hfdataset)["train"]["caption"]
        self.processor = processor

    def __len__(self) -> int:

        return len(self.images)

    def __getitem__(self, idx):
        # if not self.augmentations:
        #     image = self.processor(self.images[idx].convert("RGB"))["pixel_values"][0]
        # else:
        #     image = self.processor(self.augmentations(self.images[idx].convert("RGB")))["pixel_values"][0]
        # # print(image)
        # # input()
        # text = self.text_Processor(self.texts[idx])["input_ids"][0]
        # return image, text
        temp = self.processor(text=[self.texts[idx]], images=self.images[idx].convert("RGB"), return_tensors="pt", padding=True)
        return temp
