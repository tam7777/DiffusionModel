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
from Diffuser import Diffuser, show_images
from Unet import UNet
import os

device='cuda'
img_size = 28
batch_size = 128
num_timesteps = 1000
epochs = 10
lr = 1e-3

weight_dir = './MNIST_weight'
weight_path = os.path.join(weight_dir, 'model_weights.pth')

preprocess = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(root='/raid/miki/', download=False, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

diffuser = Diffuser(device=device)
model = UNet(num_labels=10)
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

# generate samples
images, labels = Diffuser.sample(model)
show_images(images, labels)