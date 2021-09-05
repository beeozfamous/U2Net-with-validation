import torch
from torch.utils.tensorboard import SummaryWriter
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="9"
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os

from tqdm import tqdm

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset
from data_loader import RandomGrayscale
from data_loader import RandomColorJitter
from data_loader import RandomGaussianSmoothing
from data_loader import RandomRotation
from model import U2NET
from model import U2NETP

import wandb
import logging

# ------- 0. define evaluate function --------

def evaluate(net, dataloader, writer):
    net.eval()

    for i, data in enumerate(tqdm(dataloader)):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v, i )
        writer.add_scalar("Loss/val", loss, epoch)
        writer.add_scalar("Loss2/val", loss2, epoch)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss
    net.train()
    return loss / num_val_batches, loss2 / num_val_batches

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v , i):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	if i % 20 == 0 :
		print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss

def MAE(imageA,imageB):
    mae = np.sum(np.absolute((imageB.astype("float") - imageA.astype("float")))
    return mae




# ------- 2. set the directory of training dataset --------

#-------metric & path----------
model_name = 'unet_45k_augment' #'u2netp'

data_dir = '../'
image_dir = 'coco/unlabeled2017/'
label_dir = 'coco/mask/'
model_dir = data_dir + "saved_models/"

image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

epoch_num = 100000
batch_size_train = 12
batch_size_val = 1
train_num = 0
val_percent = 0.3
learning_rate=0.0001
save_checkpoint=True
amp=False
img_scale=1

#---------end metric & path---------


def get_label(name_list,data_dir,label_dir):
    
    dataset = []
    for img_path in name_list:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]

        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]
        dataset.append(data_dir + label_dir + imidx +'_mask'+ label_ext)
    return dataset




all_img_name_list = glob.glob(data_dir + image_dir + '*' + image_ext)

n_val = int(len(all_img_name_list) * val_percent)
n_train = len(all_img_name_list) - n_val

tra_img_name_list, val_img_name_list = random_split(all_img_name_list, [n_train, n_val], generator=torch.Generator().manual_seed(0))

tra_lbl_name_list=get_label(tra_img_name_list,image_dir,label_dir)
val_lbl_name_list=get_label(val_img_name_list,image_dir,label_dir)


print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

print("---")
print("validation images: ", len(val_img_name_list))
print("validation labels: ", len(val_lbl_name_list))
print("---")



#------- 2.1 set the directory of validate dataset --------



train_salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        #RandomCrop(288),
        RandomGrayscale(0.1),
        #RandomColorJitter(0.3, brightness=(0.3,1.5), contrast=(0.3,1.5), saturation=(0.5,1.5), hue=(-0.2,0.2)),
        #RandomGaussianSmoothing(0.1),
        ToTensorLab(flag=0),
        #RandomRotation(40)        
    ])
)

val_salobj_dataset = SalObjDataset(
    img_name_list=val_img_name_list,
    lbl_name_list=val_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        #RandomCrop(288),
        RandomGrayscale(0.1),
        #RandomColorJitter(0.3, brightness=(0.3,1.5), contrast=(0.3,1.5), saturation=(0.5,1.5), hue=(-0.2,0.2)),
        #RandomGaussianSmoothing(0.1),
        ToTensorLab(flag=0),
        #RandomRotation(40)        
    ])
)

train_salobj_dataloader = DataLoader(train_salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=15)
val_salobj_dataloader = DataLoader(val_salobj_dataset, batch_size=batch_size_val, shuffle=False, num_workers=1)

#---------2.2. Create Logging-----

 # (Initialize logging)
experiment = wandb.init(project='U2-Net', resume='allow', anonymous='must')
experiment.config.update(dict(epochs=epoch_num, batch_size=batch_size_train, learning_rate=learning_rate,
                              val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))
logging.info(f'''Starting training:
    Epochs:          {epochs}
    Batch size:      {batch_size}
    Learning rate:   {learning_rate}
    Training size:   {n_train}
    Validation size: {n_val}
    Checkpoints:     {save_checkpoint}
    Device:          {device.type}
    Images scaling:  {img_scale}
    Mixed Precision: {amp}
''')

# ------- 3. define model --------

net = U2NET(3, 1)
#net.load_state_dict(torch.load(model_dir + model_name+"_bce_epoch_150_train_0.357947_tar_0.038307.pth"))
if torch.cuda.is_available():
    net.cuda()
    
    
# ------- 4. define optimizer --------


print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# # ------- 5. training process --------


print("---start training...")
                 
                 
ite_num = 0
best_loss = 99999.99
best_tar_loss = 99999.99
val_frq = 1000 # save the model every 2000 iterations






writer = SummaryWriter()
for epoch in range(0, epoch_num):
    net.train()
    
    for i, data in enumerate(tqdm(salobj_dataloader)):
        ite_num = ite_num + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v, i )
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss2/train", loss2, epoch)
                 
        experiment.log({
                    'train Loss fuse': loss.data.item(),
                    'train Sum loss all layer': loss2.data.item(),
                    'step': ite_num,
                    'epoch': epoch
                })

        loss.backward()
        optimizer.step()

        # # print statistics
        #running_loss += loss.data.item()
        #running_tar_loss += loss2.data.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss


        if ite_num % val_frq == 0:
            
            
            
            histograms = {}
            for tag, value in net.named_parameters():
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

            val_loss,val_loss2 = evaluate(net, val_salobj_dataloader,writer)
                 
            logging.info('Validation Loss fuse score: {}'.format(val_loss.data.item()))
            logging.info('Validation Sum loss all layer score: {}'.format(val_loss2.data.item()))
                 
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation Loss fuse': val_loss.data.item(),
                'validation sum loss all layer': val_loss2.data.item(),
                'step': global_step,
                'epoch': epoch,
                **histograms
            })
                 
            if val_loss.data.item()<best_loss:
                 
                 best_loss=val_loss.data.item()
                 best_tar_loss=val_loss2.data.item()
                 
                 torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, val_loss.data.item(), val_loss2.data.item()))
                 net.train() 
            elif val_loss2.data.item()<best_tar_loss:
                 
                 best_loss=val_loss.data.item()
                 best_tar_loss=val_loss2.data.item()
                 
                 torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, val_loss.data.item(), val_loss2.data.item()))
                 net.train()  
            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, val_loss.data.item(),val_loss2.data.item()))
            
            
            
            
            
            
            
            
            