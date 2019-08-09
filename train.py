import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from datasets.data import Dataset
from networks.FCN import FCN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dir = "data/train"

shuffle = True
batch_size = 8
epochs = 5
lr = 0.02
is_deconv = True
fcn_loss = nn.BCELoss()

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
    for i, (image, mask) in tqdm.tqdm(enumerate(dataloader)):
        input_batch = image.to(device=device, dtype=torch.float32)
        label_batch = image.to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        output_batch = network(input_batch)
        loss = criterion(output_batch, label_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy(output_batch, label_batch)
        counter += 1
        if counter % 20 == 0:
            print("epoch {}/{}\tbatch {}/{}\tloss {:.5f}\taccuracy {:.5f} %".format(epoch, epochs, i+1, len(dataloader), total_loss / counter,total_accuracy/counter*100))
