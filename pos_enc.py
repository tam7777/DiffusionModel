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
    
    else:
        raise ValueError('Unknown dataset')

    return img_size, dataset, in_ch, num_labels