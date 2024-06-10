import os

import datasets

dataset = datasets.load_dataset("jiviai/xray_caption_conv")

# train_dataset = dataset["train"]
val_dataset = dataset["validation"]
del dataset
train_path = "/home/jayant/Desktop/jivi/siglip/data/train"
val_path = "/home/jayant/Desktop/jivi/siglip/data/val"

# for image, caption, id in zip(train_dataset["image"], train_dataset["caption"], train_dataset["id"]):
#     image.save(os.path.join(train_path, id))
#     with open(os.path.join(train_dataset, os.path.basename(id) + ".txt"), "w") as f:
#         f.write(caption)
#         f.close()

for image, caption, id in zip(
    val_dataset["image"], val_dataset["caption"], val_dataset["id"]
):
    image.save(os.path.join(val_path, id))
    print(type(image), type(caption), type(id))
    with open(os.path.join(val_path, os.path.basename(id) + ".txt"), "w") as f:
        f.write(caption)
        f.close()
