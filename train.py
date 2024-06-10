from transformers import AutoProcessor, AutoModel, AutoTokenizer,AutoImageProcessor
import glob,os
from dataloader import multiclassdataset
from torch.utils.data import DataLoader
import torch
from freeze_layers import freeze_layer
from torch.optim import AdamW
from tqdm.auto import tqdm
import argparse

# BATCH_SIZE=8
# save_path='/home/jayant/Desktop/jivi/siglip/trained_model'
# unfreeze_layer=4


from argument import parse_args

args=parse_args()

model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
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
freeze_layer(model.vision_model,args.unfreeze_layer)
freeze_layer(model.text_model,args.unfreeze_layer)
# print(list(model.vision_model.children()))

optimizer = AdamW(model.parameters(), lr=5e-5)

losses = AverageMeter()

list_image_path = glob.glob('/') 
list_txt = ''
dataset = multiclassdataset(list_image_path,list_txt,processor,tokenizer)
train_dataloader = DataLoader(dataset,batch_size = args.batchsize,shuffle=True) #Define your own dataloader

model.train()

for epoch in range(10):  # loop over the dataset multiple times
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # get the inputs;
        pixel_values, labels = batch

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(
            pixel_values=pixel_values.to(device),
            labels=labels.to(device),
        )

        # calculate gradients
        loss = outputs.loss
        losses.update(loss.item(), pixel_values.size(0))
        loss.backward()

        # optimization step
        optimizer.step()

        if idx % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, idx, len(train_dataloader), loss=losses))
            
    save_model=os.path.join(args.save_path,f's{epoch}_loss_{loss.avg:.4f}_loss_val_{loss.val:.4f}.pth')
    torch.save(model.state_dict(), save_model)
