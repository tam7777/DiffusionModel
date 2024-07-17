import math
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from Diffuser import Diffuser
from Unet import UNet
import os

device='cuda'

batch_size = 128
num_timesteps = 1000
epochs = 1
lr = 1e-3

preprocess = transforms.ToTensor()

data='CIFAR'
if data=='CIFAR':
    img_size=32
    weight_dir = './CIFAR_weight'
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.CIFAR10(root='/raid/miki/', download=False, transform=preprocess)
    in_ch=3
    num_labels=10
elif data=='MNIST':
    img_size = 28
    weight_dir = './MNIST_weight'
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.MNIST(root='/raid/miki/', download=False, transform=preprocess)
    in_ch=1
    num_labels=10
elif data=="Celeb":
    img_size=256
    weight_dir='./Celeb_weight'
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.ImageFolder(root='/raid/miki/', download=True, transform=preprocess)
    in_ch=3
    num_labels=None

weight_path = os.path.join(weight_dir, 'model_weights.pth')

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

diffuser = Diffuser(device=device)
model = UNet(in_ch=in_ch, num_labels=num_labels)
model.to(device)
optimizer = Adam(model.parameters(), lr=lr)

# Load weights if they exist
if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path))
    print(f"Loaded weights from {weight_path}")

losses = []
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    # generate samples every epoch ===================
    #images, labels = diffuser.sample(model)
    #show_images(images, labels)
    # ================================================

    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()
        x = images.to(device)
        labels = labels.to(device)
        t = torch.randint(1, num_timesteps+1, (len(x),), device=device)

        x_noisy, noise = diffuser.add_noise(x, t)
        noise_pred = model(x_noisy, t, labels)
        loss = F.mse_loss(noise, noise_pred)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f'Epoch {epoch} | Loss: {loss_avg}')

    # Save weights every epoch
    torch.save(model.state_dict(), weight_path)

# plot losses
loss_plot_path = os.path.join(weight_dir, 'loss_plot.png')
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(loss_plot_path)

cifar_labels = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck"
}

def show_images(images, labels=None, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, i + 1)
            plt.imshow(images[i], cmap='gray')
            if labels is not None:
                if data=='CIFAR':  ax.set_xlabel(cifar_labels[int(labels[i].item())])          
                else:
                    ax.set_xlabel(labels[i].item())
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            i += 1
    plt.tight_layout()
    out_path = os.path.join(weight_dir, 'Output_image')
    plt.savefig(out_path)

# generate samples
images, labels = diffuser.sample(model, x_shape=(20, in_ch, img_size, img_size))

show_images(images, labels)