from torch.utils.data import Dataset
import numpy as np
import cv2
from PIL import Image

class multiclassdataset(Dataset):
    def __init__(self,list_image_path,list_txt,image_processor,text_Processor,transform):
        self.augmentations=transform
        self.images=list_image_path
        self.texts=list_txt
        self.processor = image_processor
        self.text_Processor=text_Processor
    def __len__(self) -> int:
    
        return len(self.images)
    
    def __getitem__(self, idx) -> tuple[np.ndarray,np.ndarray]:
        image=self.processor(Image.open(self.images[idx]))
        text=self.text_Processor(self.texts[idx])

        return image,text