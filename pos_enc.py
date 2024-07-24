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

class SCINDataset(Dataset):
    def __init__(self, img_dir, cases_csv, labels_csv, transform=None):
        self.img_dir = img_dir
        self.cases_metadata = pd.read_csv(cases_csv, dtype={'case_id': str})
        self.labels_metadata = pd.read_csv(labels_csv, dtype={'case_id': str})
        self.metadata = pd.merge(self.cases_metadata, self.labels_metadata, on='case_id')

        # Combine race and ethnicity columns into a single label
        self.metadata['race_ethnicity'] = self.metadata.apply(self.combine_race_ethnicity, axis=1)
        self.metadata['image_path'] = self.metadata['image_1_path'].apply(self.get_full_image_path)
        self.metadata = self.metadata[self.metadata['image_path'].apply(os.path.exists)]
        
        self.transform = transform
        self.label_map = {label: idx for idx, label in enumerate(self.metadata['race_ethnicity'].unique())}

    def combine_race_ethnicity(self, row):
        races = ['race_ethnicity_american_indian_or_alaska_native', 'race_ethnicity_asian', 
                 'race_ethnicity_black_or_african_american', 'race_ethnicity_hispanic_latino_or_spanish_origin',
                 'race_ethnicity_middle_eastern_or_north_african', 'race_ethnicity_native_hawaiian_or_pacific_islander',
                 'race_ethnicity_white', 'race_ethnicity_other_race']
        combined = [race[len("race_ethnicity_"):] for race in races if row[race] == 'YES']
        return ",".join(combined) if combined else 'unspecified'

    def get_full_image_path(self, img_path):
        return os.path.join(self.img_dir, os.path.basename(img_path))
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        label = row['race_ethnicity']

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.label_map[label], dtype=torch.long)

        return image, label

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
    
    elif data=='scin':
        img_size = 224
        preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        img_dir = '/raid/miki/scin/dx-scin-public-data/dataset/images'
        cases_csv = '/raid/miki/scin/dx-scin-public-data/dataset/scin_cases.csv'
        labels_csv = '/raid/miki/scin/dx-scin-public-data/dataset/scin_labels.csv'
        
        dataset = SCINDataset(img_dir=img_dir, cases_csv=cases_csv, labels_csv=labels_csv, transform=preprocess)
        
        in_ch = 3
        num_labels = len(dataset.metadata['race_ethnicity'].unique())

    else:
        raise ValueError('Unknown dataset')

    return img_size, dataset, in_ch, num_labels

def save_losses_to_file(losses, file_path):
    # Create the file if it does not exist and append the losses
    with open(file_path, 'a') as f:
        for loss in losses:
            f.write(f'{loss}\n')

import matplotlib.pyplot as plt
import os
import PIL
def delete_file(file_path, new_file_path):
    if os.path.exists(file_path):
        os.rename(file_path, new_file_path)
        print(f'File {file_path} has been changed to {new_file_path}.')
    else:
        print(f'File {file_path} does not exist.')

import os

def get_number_of_epochs(weights_dir='./weight', loss_file='./losses.txt'):
    # Get the list of weight files
    weight_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
    
    # Extract epoch numbers from the filenames
    epochs = []
    for f in weight_files:
        epoch = int(f.split('_')[0])
        epochs.append(epoch)

    
    # Determine the latest epoch
    latest_epoch = max(epochs) if epochs else 0
    if os.path.exists(loss_file):
    # Read the loss file
        with open(loss_file, 'r') as f:
            loss_lines = f.readlines()
        
        # Adjust the loss file content to match the latest epoch
        adjusted_loss_lines = loss_lines[:latest_epoch]
        
        # Write the adjusted loss lines back to the file
        with open(loss_file, 'w') as f:
            f.writelines(adjusted_loss_lines)
    else:
        print(f'File {loss_file} does not exist. Propably there was an error whilst training and could not save loss.txt file')
    
    print(f"Your latest epoch_is: {latest_epoch}")

    return latest_epoch

if __name__ == "__main__":
    dir="./HAM_weight_dit/weight/"
    loss="./HAM_weight_dit/losses.txt"
    adjust_loss_file(dir, loss)

    import sys; sys.exit(0)

    img_size, scin_dataset, in_ch, num_labels = get_data("scin")
    print(f"Number of samples in the dataset: {len(scin_dataset)}")
    scin_loader = DataLoader(scin_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Example of iterating through the data loader
    for images, labels in scin_loader:
        print(type(labels))
        if images is None or labels is None:
            continue  # Skip missing file entries
        # Your training loop here
        pass