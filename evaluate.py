import os
import torch
import inspect

# Export include path
os.environ['C_INCLUDE_PATH'] = '/home/miki/Python-3.10.12/include/python3.10'

#torchrun --standalone --nproc_per_node=8 main.py

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
import torch._dynamo
#torch._dynamo.config.suppress_errors = True

import warnings
warnings.simplefilter('ignore')

# Load configuration
with open('config.yaml', 'rb') as f:
    yml = yaml.safe_load(f)

device=yml['Main']['device']
batch_size = yml['Main']['batch_size']
num_timesteps = yml['Main']['num_timesteps']
epochs = yml['Main']['epochs']
lr = float(yml['Main']['lr'])
model_type = yml['Main']['model_type']
data = yml['Main']['data']
model_size=yml['Main']['size']

from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Use DDP (Distributed Data Parallel)
ddp = int(os.environ.get('RANK', -1)) != -1 
if ddp:
    assert torch.cuda.is_available(), "CUDA not available"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])  # Global rank of the process
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # Local rank of the process
    ddp_world_size = int(os.environ['WORLD_SIZE'])  # Total number of processes
    device = f'cuda:{ddp_local_rank}'  # Assign unique device for each process
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # True if this is the master process
else:
    # Non-DDP
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda"
    print("Running in non-DDP mode")
device_dtype='cuda'

# Get data
img_size, dataset, in_ch, num_labels = get_data(data)

if model_size!='s':
    weight_dir = f"./{data}_weight_{model_type}_{model_size}"
else:
    weight_dir = f"./{data}_weight_{model_type}"

weight_path = os.path.join(weight_dir, 'model_weights.pth')

if model_size!='s':
    gen_dir=f"/raid/miki/FID_{data}/graph/{model_type}_m"
else:
    gen_dir=f"/raid/miki/FID_{data}/graph/{model_type}"

gen_images_dir=os.path.join(gen_dir, 'generated')
real_images_dir=os.path.join(gen_dir, 'original')

dataloader = DataLoader(
    dataset,
    batch_size=batch_size // ddp_world_size,
    shuffle=False,
    sampler=DistributedSampler(dataset, num_replicas=ddp_world_size, rank=ddp_rank,shuffle=True),
    num_workers=4,
    pin_memory=True
)
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
model=torch.compile(model)
model = DDP(model, device_ids=[ddp_local_rank])
if master_process:print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


if os.path.exists(weight_path):
    try:
        model.load_state_dict(torch.load(weight_path))
        if master_process:print(f"Loaded weights from {weight_path}")
    except:
        if master_process:print("Weights not load")
        import sys; sys.exit()
else:
    if master_process:print(f"Can not find {weight_path}")
    import sys; sys.exit()


losses = []

if master_process:

    HAM_labels = {0:'nv', 1:'mel', 2:'bkl', 3:'bcc', 4:'akiec', 5:'vasc', 6:'df'}

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
                    if data == 'HAM':  
                        ax.set_xlabel(HAM_labels[int(labels[i].item())])
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
        images_latent, labels = diffuser.sample(model, x_shape=(num_samples, latent_dim, latent_size, latent_size), num_labels=num_labels)
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


    fid_metrics = torch_fidelity.calculate_metrics(
        input1=real_images_dir,
        input2=gen_images_dir,
        cuda=True,
        fid=True,
        precision=True,
        recall=True
    )

    print(f"FID: {fid_metrics['frechet_inception_distance']}")

    # Safely print precision and recall if they exist in the dictionary
    if 'precision' in fid_metrics:
        print(f"Precision: {fid_metrics['precision']}")
    else:
        print("Precision not available in the results")

    if 'recall' in fid_metrics:
        print(f"Recall: {fid_metrics['recall']}")
    else:
        print("Recall not available in the results")

if ddp:destroy_process_group()
