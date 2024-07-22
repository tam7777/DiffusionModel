import os
import torch

# Verify which GPU is being used
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(device)
print(f"Using GPU: {torch.cuda.current_device()} with {gpu_name}")

import math
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm
from Diffuser import Diffuser
from Unet import UNet
from dit import DiT
from transformer import Beta_DiT
import yaml
from pos_enc import get_data
from torchvision.utils import save_image
from torchvision import transforms
import PIL
import torch_fidelity
from diffusers import AutoencoderKL

# Load configuration
with open('config.yaml', 'rb') as f:
    yml = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = yml['Main']['batch_size']
num_timesteps = yml['Main']['num_timesteps']
epochs = yml['Main']['epochs']
lr = float(yml['Main']['lr'])
model_type = yml['Main']['model_type']
data = yml['Main']['data']

# Get data
img_size, dataset, in_ch, num_labels = get_data(data)
weight_dir = f"./{data}_weight_{model_type}"
weight_path = os.path.join(weight_dir, 'model_weights.pth')

gen_images_dir=os.path.join(weight_dir, 'generated')
real_images_dir=os.path.join(weight_dir, 'original')

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
diffuser = Diffuser(device=device)

# Model selection
if model_type == "unet":
    model = UNet(in_ch=in_ch, time_embed_dim=yml['Unet']['time_embed_dim'], num_labels=num_labels)
elif model_type == "dit":
    model = DiT(input_size=img_size, patch_size=2, in_channels=in_ch, hidden_size=yml['DiT']['n_embd'], 
                depth=yml['DiT']['n_layer'], num_heads=yml['DiT']['n_head'], num_classes=num_labels)
elif model_type == "dit_beta":
    model = Beta_DiT(in_ch=in_ch, img_size=img_size, num_labels=num_labels)
else:
    raise ValueError("Model type should be either 'unet' or 'dit'")


model.to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

optimizer = AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Load weights if they exist
if os.path.exists(weight_path):
    try:
        model.load_state_dict(torch.load(weight_path))
        print(f"Loaded weights from {weight_path}")
    except:
        print("Weights not load")

torch.set_float32_matmul_precision('high')

losses = []
for epoch in range(epochs):
    model.train()
    loss_sum = 0.0
    cnt = 0

    for images, labels in tqdm(dataloader):
        optimizer.zero_grad()
        x = images.to(device)
        labels = labels.to(device)
        t = torch.randint(1, num_timesteps + 1, (len(x),), device=device)

        x_noisy, noise = diffuser.add_noise(x, t)
        noise_pred = model(x_noisy, t, labels)
        loss = F.mse_loss(noise_pred, noise)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    scheduler.step()
    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(f'Epoch {epoch} | Loss: {loss_avg}')

    torch.save(model.state_dict(), weight_path)

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
    out_path = os.path.join(weight_dir, 'output_image')
    plt.savefig(out_path)

# generate samples
num_samples = yml['Main']['num_samples']
model.eval()
with torch.no_grad():
    images, labels = diffuser.sample(model, x_shape=(num_samples, in_ch, img_size, img_size), num_labels=num_labels)

show_images(images, labels)

for idx, (image, label) in enumerate(dataset):
    save_image(image, os.path.join(real_images_dir, f'real_{idx}.png'))
    if idx+1>=num_samples: break

# Save generated images
# Ensure all images are of the same size
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])
for i, img in enumerate(images):
    if isinstance(img, PIL.Image.Image):
        img = transform(img)
    save_image(img, os.path.join(gen_images_dir, f"gen_{i}.png"))


fid_score = torch_fidelity.calculate_metrics(
    input1=real_images_dir,
    input2=gen_images_dir,
    cuda=True,
    fid=True
)

print(f"FID: {fid_score['frechet_inception_distance']}")
