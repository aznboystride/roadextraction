import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from datasets.data import Dataset
from networks.FCN import FCN
from sys import argv
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = [0, 1, 2, 3]

# Jaccard - Intersection Over Union
def dice_coeff(outputs, labels):
    intersection = torch.cuda.torch.sum(outputs * labels)
    sum = torch.cuda.torch.sum(outputs + labels)
    dice = 2. * intersection / sum
    return dice.mean()

def dice_loss(outputs, labels):
    return 1 - dice_coeff(outputs, labels)

def bce_dice(outputs, labels):
    return nn.BCELoss()(outputs, labels) + dice_loss(outputs, labels)

def split_loaders(dataset, shuffle=True, valsplit=0.2):
    size = len(dataset)
    indices = list(range(size))
    split = int(np.floor(size * valsplit))
    if shuffle:
        np.random.shuffle(indices)
    train_indices = indices[:split]
    valid_indices = indices[split:]
    train_sampler, valid_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(valid_indices)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler), torch.utils.data.DataLoader(dataset, batch_size=len(dataset), sampler=valid_sampler)

def valid(model, loader, batch_size):
    total_loss = 0
    total_acc = 0
    counter = 1
    with torch.no_grad():
        for (input_batch, label_batch) in loader:
            input_batch = input_batch
            label_batch = label_batch
            optimizer.zero_grad()
            output_batch = model(input_batch)    
            loss = criterion(output_batch, label_batch)
            total_loss += loss.item()
            total_acc += accuracy(output_batch, label_batch)
            if counter % 20 == 0:
                #print("data {}/{}\tloss {:.5f}\tacc {:.5f}".format(counter*batch_size, batch_size*len(loader), total_loss / counter, total_acc / counter)) 
                print("\t\tdata {}/{}\tloss {:.5f}\tacc {:.5f}".format(counter * batch_size, batch_size*len(train_loader), total_loss / counter, total_acc / counter)) 
            counter += 1

accuracy = lambda x, y: dice_coeff(x, y)

train_dir = "data/train"

shuffle = True
batch_size = int(argv[1])
epochs = 5
lr = 0.001
is_deconv = True
fcn_loss = bce_dice

network = FCN(is_deconv=is_deconv).cuda()
network = nn.DataParallel(network)
criterion = fcn_loss
optimizer = optim.Adam(network.parameters(), lr=lr)

image_filenames = []

for filename in os.listdir(train_dir):
    image_filenames.append(filename)


dataset = Dataset(train_dir, image_filenames)
train_loader, valid_loader = split_loaders(dataset)

# Training
d = 0
for epoch in range(1, epochs+1):
    total_loss = 0
    total_accuracy = 0
    print("Epoch {}/{}:".format(epoch, epochs))
    for counter, (image, mask) in enumerate(train_loader):
        if counter != 0 and counter % 20 == 0:
            print("\t\tdata {}/{}\tloss {:.5f}\tacc {:.5f}".format(counter * batch_size, batch_size*len(train_loader), total_loss / counter, total_accuracy / counter)) 
        input_batch = image.to(device=device[d], dtype=torch.float32)
        d +=1
        label_batch = mask.to(device=device[d], dtype=torch.float32)
        d +=1
        if d > 3:
            d = 0
        optimizer.zero_grad()
        output_batch = network(input_batch)
        loss = criterion(output_batch, label_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy(output_batch, label_batch)
            # print("epoch {}/{}\tbatch {}/{}\tloss {:.5f}\taccuracy {:.5f} ".format(epoch, epochs, i+1, len(train_loader), total_loss / counter,total_accuracy/counter))
    print("\nValidating...")
    valid(network, valid_loader)
