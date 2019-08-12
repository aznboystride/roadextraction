from networks.FCN import FCN
import os
import torch
import torchvision.transforms as T
from PIL import Image

weight_path = "fcnweight"

train_dir = "valid"

# Load model
model = FCN(True).cuda()

# Load weights
model.load_state_dict(torch.load(weight_path))
model.eval() # Set evaluation mode

def eval_image(path):
    with torch.no_grad():
        image = Image.open(path)
        

