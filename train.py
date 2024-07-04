import argparse
import glob
import os

import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoImageProcessor, AutoModel, AutoProcessor,
                          AutoTokenizer)

from argument import parse_args
from dataloader import multiclassdataset
from freeze_layers import freeze_layer
import wandb
wandb.init( project="sig_train")
# BATCH_SIZE=8
# save_path='/home/jayant/Desktop/jivi/siglip/trained_model'
# unfreeze_layer=4



args = parse_args()
wandb.config = {"epochs": args.epochs,
                "batch_size":args.batch_size,
                "unfreezed_vision_encoder_layer":args.vision_encoder,
                "unfreezed_vision_encoder_layer":args.text_encoder,
                "learning_rate": args.lr
                }



model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
auto_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# freeze_layer(model.vision_model, args.unfreeze_layer)
# freeze_layer(model.text_model, args.unfreeze_layer)
# print(list(model.vision_model.children()))

losses = AverageMeter()

# list_image_path = glob.glob('/')
# list_txt = ''
hf_dataset = "jiviai/RSNA_Refined"
# hf_dataset = "jiviai/xray_test"
# dataset = multiclassdataset(hf_dataset, processor, tokenizer)


model.train()

for layer in model.parameters():
        layer.requires_grad = False
if args.vision_encoder >0:
    layers=int(args.vision_encoder)*-1
    if args.vision_encoder:
        for layer in model.vision_model.encoder.layers[layers:]:
            layer.requires_grad_(True)
        model.vision_model.post_layernorm.requires_grad_(True)
        model.vision_model.head.requires_grad_(True)
        
if args.text_encoder:
    layers=int(args.text_encoder)*-1
    for layer in model.text_model.encoder.layers[layers:]:
        layer.requires_grad_(True)
    model.text_model.final_layer_norm.requires_grad_(True)
    model.text_model.head.requires_grad_(True)

# for name,p in model.named_parameters():
#     if p.requires_grad:
#         print(name)
# print('check')
dataset = multiclassdataset(hf_dataset, auto_processor)
train_dataloader = DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
)  # Define your own dataloader
optimizer = AdamW(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer, T_max=len(dataset) / args.batch_size * args.epochs
)
for epoch in tqdm(range(args.epochs),total=args.epochs):  # loop over the dataset multiple times
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # # get the inputs;
        # zero the parameter gradients
        optimizer.zero_grad()
        batch["pixel_values"] = batch["pixel_values"]
        batch['return_loss']=True
        inputs = batch
        # batch.to(device)
        # forward pass
        outputs = model(
            **inputs
            )
        # calculate gradients
        loss = outputs.loss
        losses.update(loss.item(), batch["pixel_values"].size(0))
        wandb.log({"loss": loss.item(),"epoch":epoch})


        loss.backward()

        # optimization step
        optimizer.step()
        scheduler.step()
        # print(f"running loss : {losses.avg}")
        if idx % 100 == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch, idx, len(train_dataloader), loss=losses
                )
            )

    save_model = os.path.join(
        args.save_path, f"epoch_{epoch}.pth"
    )
    torch.save(model.state_dict(), save_model)
