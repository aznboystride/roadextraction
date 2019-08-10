import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from datasets.data import Dataset
from networks.FCN import FCN
from sys import argv

# Jaccard - Intersection Over Union
def dice_coeff(outputs, labels):
    intersection = torch.sum(outputs * labels)
    sum = torch.sum(outputs + labels)
    dice = 2. * intersection / sum
    return dice.mean()

def dice_loss(outputs, labels):
    return 1 - dice_coeff(outputs, labels)

def bce_dice(outputs, labels):
    return nn.BCELoss()(outputs, labels) + dice_loss(outputs, labels)

accuracy = lambda x, y: dice_coeff(x, y)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dir = "data/train"

shuffle = True
batch_size = int(argv[1])
epochs = 5
lr = 0.001
is_deconv = True
fcn_loss = bce_dice

network = FCN(is_deconv=is_deconv).cuda()
criterion = fcn_loss
optimizer = optim.Adam(network.parameters(), lr=lr)

image_filenames = []

for filename in os.listdir(train_dir):
    image_filenames.append(filename)


dataset = Dataset(train_dir, image_filenames)
dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle)

# Training
for epoch in range(1, epochs+1):
    total_loss = 0
    total_accuracy = 0
    counter = 0
    for i, (image, mask) in tqdm.tqdm(enumerate(dataloader)):
        input_batch = image.to(device=device, dtype=torch.float32)
        label_batch = mask.to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        output_batch = network(input_batch)
        loss = criterion(output_batch, label_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy(output_batch, label_batch)
        counter += 1
        if counter % 20 == 0:
            print("epoch {}/{}\tbatch {}/{}\tloss {:.5f}\taccuracy {:.5f} ".format(epoch, epochs, i+1, len(dataloader), total_loss / counter,total_accuracy/counter))
