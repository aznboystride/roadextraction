import os
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

best = 0

train_log = "train.log"

weight_path = "fcnweight"

train_dir = "data/train"

shuffle = True

batch_size = 16

epochs = 5

lr = 0.001

is_deconv = True

# Jaccard - Intersection Over Union
def dice_coeff(outputs, labels):
    with torch.no_grad():
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
    valid_indices = indices[:split]
    train_indices = indices[split:]
    train_sampler, valid_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(valid_indices)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler), torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

def valid(model, loader, batch_size):
    global best
    total_loss = 0
    total_acc = 0
    counter = 1
    model.eval()
    with torch.no_grad():
        for (input_batch, label_batch) in loader:
            if total_acc / counter > best:
                print("Beat Current Best: {:.5f} -> {:.5f}".format(best, total_acc / counter))
                print("Beat Current Best: {:.5f} -> {:.5f}".format(best, total_acc / counter), file=open(train_log, 'a'))
                best = total_acc / counter
                torch.save(network.state_dict(), weight_path)
                
            input_batch = input_batch.cuda()
            label_batch = label_batch.cuda()
            optimizer.zero_grad()
            output_batch = model(input_batch)    
            loss = criterion(output_batch, label_batch)
            total_loss += loss.item()
            total_acc += accuracy(output_batch, label_batch)
            if counter % 20 == 0:
                #print("data {}/{}\tloss {:.5f}\tacc {:.5f}".format(counter*batch_size, batch_size*len(loader), total_loss / counter, total_acc / counter)) 
                print("\t\tdata {}/{}\tloss {:.5f}\tacc {:.5f}".format(counter * batch_size, batch_size*len(loader), total_loss / counter, total_acc / counter)) 
                print("\t\tdata {}/{}\tloss {:.5f}\tacc {:.5f}".format(counter * batch_size, batch_size*len(loader), total_loss / counter, total_acc / counter), file=open(train_log, 'a')) 
                counter += 1
    model.train()

accuracy = lambda x, y: dice_coeff(x, y)

fcn_loss = bce_dice

network = FCN(is_deconv=is_deconv).cuda()
network = nn.DataParallel(network, device_ids=[0,1,2,3])
criterion = fcn_loss
optimizer = optim.Adam(network.parameters(), lr=lr)

image_filenames = []

for filename in os.listdir(train_dir):
    image_filenames.append(filename)


dataset = Dataset(train_dir, image_filenames)

# Training
for epoch in range(1, epochs+1):
    total_loss = 0
    total_accuracy = 0
    train_loader, valid_loader = split_loaders(dataset)
    print("Epoch {}/{}:".format(epoch, epochs))
    print("Epoch {}/{}:".format(epoch, epochs), file=open(train_log, 'a'))
    for counter, (image, mask) in enumerate(train_loader):
        if counter != 0 and counter % 20 == 0:
            print("\t\tdata {}/{}\tloss {:.5f}\tacc {:.5f}".format(counter * batch_size, batch_size*len(train_loader), total_loss / counter, total_accuracy / counter))
            print("\t\tdata {}/{}\tloss {:.5f}\tacc {:.5f}".format(counter * batch_size, batch_size*len(train_loader), total_loss / counter, total_accuracy / counter), file=open(train_log, 'a')) 
        input_batch = image.cuda()
        label_batch = mask.cuda()
        optimizer.zero_grad()
        output_batch = network(input_batch)
        loss = criterion(output_batch, label_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy(output_batch, label_batch)
        # torch.cuda.empty_cache() 
        del image, mask, output_batch, label_batch, input_batch, loss
            # print("epoch {}/{}\tbatch {}/{}\tloss {:.5f}\taccuracy {:.5f} ".format(epoch, epochs, i+1, len(train_loader), total_loss / counter,total_accuracy/counter))
       
    del train_loader 
    print("\nValidating...")
    print("\nValidating...", file=open(train_log, 'a'))
    valid(network, valid_loader, batch_size)

