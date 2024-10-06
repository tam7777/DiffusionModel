import os
import torch
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import transforms
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch_fidelity
from Diffuser import Diffuser
from Unet import UNet
from dit import DiT
from transformer import Beta_DiT
from pos_enc import get_data

def strip_prefix_if_present(state_dict, prefix):
    """Strip the prefix from the state_dict keys if it exists."""
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = {}
    for key in keys:
        stripped_state_dict[key[len(prefix):]] = state_dict[key]
    return stripped_state_dict

# Load configuration
with open('config.yaml', 'rb') as f:
    yml = yaml.safe_load(f)

device = yml['Main']['device']
batch_size = yml['Main']['batch_size']
model_type = yml['Main']['model_type']
data = yml['Main']['data']
model_size = yml['Main']['size']

if model_size != 's':
    weight_dir = f"/home/miki/DiffusionModel/{data}_weight_{model_type}_{model_size}/weight"
else:
    weight_dir = f"/home/miki/DiffusionModel/{data}_weight_{model_type}/weight"

if model_size != 's':
    gen_dir = f"/raid/miki/FID_{data}/graph/{model_type}_m"
else:
    gen_dir = f"/raid/miki/FID_{data}/graph/{model_type}"

gen_images_dir = os.path.join(gen_dir, 'generated')
real_images_dir = os.path.join(gen_dir, 'original')

# Ensure directories exist
os.makedirs(gen_images_dir, exist_ok=True)
os.makedirs(real_images_dir, exist_ok=True)

# Get data
img_size, dataset, in_ch, num_labels = get_data(data)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    sampler=DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True),
    num_workers=4,
    pin_memory=True
)

diffuser = Diffuser(device=device)

# Initialize VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

latent_dim = 4
latent_size = img_size // 8

# Model selection
if model_type == "unet":
    if data != "MNIST":
        model = UNet(in_ch=latent_dim, time_embed_dim=yml['Unet']['time_embed_dim'], num_labels=num_labels)
    else:
        model = UNet(in_ch=in_ch, time_embed_dim=yml['Unet']['time_embed_dim'], num_labels=num_labels)
elif model_type == "dit":
    if data != "MNIST":
        model = DiT(input_size=latent_size, patch_size=2, in_channels=latent_dim, hidden_size=yml['DiT']['n_embd'],
                    depth=yml['DiT']['n_layer'], num_heads=yml['DiT']['n_head'], num_classes=num_labels)
    else:
        model = DiT(input_size=img_size, patch_size=2, in_channels=in_ch, hidden_size=yml['DiT']['n_embd'],
                    depth=yml['DiT']['n_layer'], num_heads=yml['DiT']['n_head'], num_classes=num_labels)
elif model_type == "dit_beta":
    if data != "MNIST":
        model = Beta_DiT(in_ch=latent_dim, img_size=latent_size, num_labels=num_labels)
    else:
        model = Beta_DiT(in_ch=in_ch, img_size=img_size, num_labels=num_labels)
else:
    raise ValueError("Model type should be either 'unet' or 'dit'")

model.to(device)
model = torch.compile(model)

# Load real images
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

for idx, (image, label) in enumerate(dataset):
    save_image(image, os.path.join(real_images_dir, f'real_{idx}.png'))
    if idx + 1 >= 100:  # Save 100 real images
        break

# List files in the directory
print("Files in weight directory:")
print(os.listdir(weight_dir))

fid_scores = []

# Iterate over the weight files every 50 epochs and compute FID
for epoch in range(0, 2476, 100):
    weight_path = os.path.join(weight_dir, f'{epoch}_weight.pth')
    if os.path.exists(weight_path):
        print(f"Processing weights for epoch {epoch}")
        state_dict = torch.load(weight_path)
        state_dict = strip_prefix_if_present(state_dict, "module.")
        model.load_state_dict(state_dict)
        model.eval()

        with torch.no_grad():
            images_latent, labels = diffuser.sample(model, x_shape=(100, latent_dim, latent_size, latent_size), num_labels=num_labels)
            if data != "MNIST":
                images = vae.decode(images_latent / 0.18215).sample

        # Ensure the images are saved correctly using save_image
        for i in range(images.size(0)):
            save_image(images[i], os.path.join(gen_images_dir, f"gen_{i}.png"), normalize=True, value_range=(-1, 1))

        fid_metrics = torch_fidelity.calculate_metrics(
            input1=real_images_dir,
            input2=gen_images_dir,
            cuda=True,
            fid=True
        )

        fid_scores.append((epoch, fid_metrics['frechet_inception_distance']))
    else:
        print(f"Could not find weight file for epoch {epoch} at path {weight_path}")

# Plot the FID scores
if fid_scores:
    epochs, fids = zip(*fid_scores)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, fids, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('FID Score')
    plt.title('FID Score vs. Epoch')
    plt.grid(True)
    plt.savefig(os.path.join(gen_dir, 'fid_scores.png'))
else:
    print("No FID scores were calculated. Check if the weight files exist and are accessible.")

print("finished")
