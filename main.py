import os
import torch

#CUDA_VISIBLE_DEVICES=7 python3 main.py

# Verify which GPU is being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
from pos_enc import get_data, save_losses_to_file, delete_file,get_number_of_epochs
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

# Initialize VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

latent_dim=4
latent_size=img_size//8

# Model selection
if model_type == "unet":
    if data!="MNIST":
        model = UNet(in_ch=latent_dim, time_embed_dim=yml['Unet']['time_embed_dim'], num_labels=num_labels)
    else:
        model = UNet(in_ch=in_ch, time_embed_dim=yml['Unet']['time_embed_dim'], num_labels=num_labels)
elif model_type == "dit":
    if data!="MNIST":
        model = DiT(input_size=latent_size, patch_size=2, in_channels=latent_dim, hidden_size=yml['DiT']['n_embd'], 
                depth=yml['DiT']['n_layer'], num_heads=yml['DiT']['n_head'], num_classes=num_labels)
    else:
        model = DiT(input_size=img_size, patch_size=2, in_channels=in_ch, hidden_size=yml['DiT']['n_embd'], 
                depth=yml['DiT']['n_layer'], num_heads=yml['DiT']['n_head'], num_classes=num_labels)    
elif model_type == "dit_beta":
    if data!="MNIST":
        model = Beta_DiT(in_ch=latent_dim, img_size=latent_size, num_labels=num_labels)
    else:
        model = Beta_DiT(in_ch=in_ch, img_size=img_size, num_labels=num_labels)
else:
    raise ValueError("Model type should be either 'unet' or 'dit'")


model.to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

optimizer = AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Load weights if they exist
loss_file_path=os.path.join(weight_dir, 'losses.txt')
exiting_epoch=0
if os.path.exists(weight_path):
    try:
        model.load_state_dict(torch.load(weight_path))
        print(f"Loaded weights from {weight_path}")
        exiting_epoch=get_number_of_epochs(loss_file_path)
    except:
        print("Weights not load")
        delete_file(loss_file_path)

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

        # Encode images to latent space
        if data != "MNIST":
            with torch.no_grad():
                x_latent = vae.encode(x).latent_dist.sample().mul_(0.18215)  # Scaling factor used in official implementation
        else:
            x_latent=x
        x_latent=x_latent.to(device)

        x_noisy, noise = diffuser.add_noise(x_latent, t)
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

    # Save weights every 10 epoch
    if epoch%10==0:
        temp_weight_path = weight_dir + f'/weight/{epoch+exiting_epoch}_weight.pth'
        torch.save(model.state_dict(), temp_weight_path)

torch.save(model.state_dict(), weight_path)

# plot losses
save_losses_to_file(losses, loss_file_path)
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

def show_images(image_dir, labels=None, rows=2, cols=10):
    fig = plt.figure(figsize=(cols, rows))
    i = 0
    for r in range(rows):
        for c in range(cols):
            img_path=image_dir+f"/generated/gen_{i}.png"
            img = PIL.Image.open(img_path)
            ax = fig.add_subplot(rows, cols, i + 1)
            plt.imshow(img)
            if labels is not None:
                if data == 'CIFAR':  
                    ax.set_xlabel(cifar_labels[int(labels[i].item())])
                else:
                    ax.set_xlabel(labels[i].item())
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            i += 1
            if i>=20: break
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, 'output_image.png'))


# generate samples
num_samples = yml['Main']['num_samples']
model.eval()
with torch.no_grad():
    images_latent, labels = diffuser.sample(model, x_shape=(20, latent_dim, latent_size, latent_size), num_labels=num_labels)
    if data!="MNIST":
        images = vae.decode(images_latent / 0.18215).sample

# Ensure the images are saved correctly using save_image
for i in range(images.size(0)):
    save_image(images[i], os.path.join(gen_images_dir, f"gen_{i}.png"), normalize=True, value_range=(-1, 1))

show_images(weight_dir, labels)

for idx, (image, label) in enumerate(dataset):
    save_image(image, os.path.join(real_images_dir, f'real_{idx}.png'))
    if idx+1 >= num_samples: break

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