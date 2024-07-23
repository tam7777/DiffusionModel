import torch

def _pos_encoding(ts, output_dim, device='cuda'):
    t,D=ts, output_dim
    v=torch.zeros(D, device=device)

    i=torch.arange(0,D,device=device)
    div_term=10000**(i/D)

    v[0::2]=torch.sin(t / div_term[0::2])
    v[1::2]=torch.cos(t / div_term[1::2])

    return v

def pos_encoding(timesteps, output_dim, device='cuda'):
    batch_size=len(timesteps)
    v=torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i]=_pos_encoding(timesteps[i], output_dim, device)

    return v

import torchvision
from torchvision import transforms
import numpy as np
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np

class HAM10000Dataset(Dataset):
    def __init__(self, img_dirs, csv_file, transform=None):
        self.img_dirs = img_dirs
        self.metadata = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_id = self.metadata.iloc[idx]['image_id']
        img_dir = self.img_dirs[0] if os.path.exists(os.path.join(self.img_dirs[0], f'{img_id}.jpg')) else self.img_dirs[1]
        img_path = os.path.join(img_dir, f'{img_id}.jpg')
        image = Image.open(img_path).convert('RGB')
        label = self.metadata.iloc[idx]['dx']

        if self.transform:
            image = self.transform(image)

        # Map the labels to integers
        label_map = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
        label = label_map[label]

        return image, label

def get_data(data):
    if data == 'CIFAR':
        img_size = 32
        preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.CIFAR10(root='/raid/miki/', download=False, transform=preprocess)
        in_ch = 3
        num_labels = 10
    elif data == 'MNIST':
        img_size = 28
        preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.MNIST(root='/raid/miki/', download=False, transform=preprocess)
        in_ch = 1
        num_labels = 10
    elif data == "ImageNet":
        img_size = 224
        preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.ImageFolder(root='/raid/miki/imagenet/ILSVRC/Data/CLS-LOC/train', transform=preprocess)
        in_ch = 3
        num_labels = 1000

        num_images = 1000
        indices = np.random.choice(len(dataset), num_images, replace=False)
        dataset = Subset(dataset, indices)
    
    elif data == 'HAM':
        img_size = 224
        preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        img_dirs = [
            '/raid/miki/skin-cancer/HAM10000_images_part_1',
            '/raid/miki/skin-cancer/HAM10000_images_part_2'
        ]
        csv_file = '/raid/miki/skin-cancer/HAM10000_metadata.csv'
        dataset = HAM10000Dataset(img_dirs=img_dirs, csv_file=csv_file, transform=preprocess)
        in_ch = 3
        num_labels = 7  # There are 7 classes in the HAM10000 dataset
    else:
        raise ValueError('Unknown dataset')

    return img_size, dataset, in_ch, num_labels

def save_losses_to_file(losses, file_path):
    # Create the file if it does not exist and append the losses
    with open(file_path, 'a') as f:
        for loss in losses:
            f.write(f'{loss}\n')

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f'File {file_path} has been deleted.')
    else:
        print(f'File {file_path} does not exist.')

def get_number_of_epochs(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            return len(lines)
    else:
        return 0