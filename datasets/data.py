import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, train_dir, image_filenames):
        self.image_filenames = image_filenames
        self.train_dir = train_dir
        self.batch_size=batch_size
        self.trf = T.Compose([T.ToTensor()])
    
    def __getitem__(self, index):
        image_path = os.path.join(self.train_dir, image_filenames[index])
        image = Image.open(image_path)
        image = T.Compose([T.ToTensor()])(image)
        mask_path = os.path.join(train_dir, image_filenames[index].split('_')[0] + "_mask.png")
        mask = self.trf(Image.open(mask_path).convert('L')) / 255
        return image, mask
    #torch.tensor(torch.from_numpy(np.rollaxis(image, 2, 0)), dtype=torch.int64), torch.tensor(torch.from_numpy(np.rollaxis(mask, 2, 0)), dtype=torch.int64)
    
    def __len__(self):
        return len(self.image_filenames)
