import torch
import yaml
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL

class Diffuser:
    def __init__(self, device):
        with open('config.yaml', 'rb') as f:
            yml=yaml.safe_load(f)

        self.num_timesteps = yml['Diffuser']['num_timesteps']
        self.device = device
        self.beta_start = yml['Diffuser']['beta_start']
        self.beta_end = yml['Diffuser']['beta_end']
        self.betas=torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, device=self.device)
        self.alphas=1-self.betas
        self.alpha_bars=torch.cumprod(self.alphas, dim=0)
        #self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

    def add_noise(self, x_0, t):
        T=self.num_timesteps
        assert (t>=1).all() and (t<=T).all()

        t_idx=t-1
        alpha_bar=self.alpha_bars[t_idx]
        alpha_bar=alpha_bar.view(alpha_bar.size(0),1,1,1)

        noise=torch.randn_like(x_0, device=self.device)
        x_t=torch.sqrt(alpha_bar)*x_0+torch.sqrt(1-alpha_bar)*noise

        return x_t, noise

    def denoise(self,model,x,t,labels):
        T=self.num_timesteps
        assert (t>=1).all() and (t<=T).all()

        t_idx=t-1
        alpha=self.alphas[t_idx]
        alpha_bar=self.alpha_bars[t_idx]
        alpha_bar_prev=self.alpha_bars[t_idx-1]

        N=alpha.size(0)
        alpha=alpha.view(N,1,1,1)
        alpha_bar=alpha_bar.view(N,1,1,1)
        alpha_bar_prev=alpha_bar_prev.view(N,1,1,1)

        model.eval()
        with torch.no_grad():
            eps=model(x,t,labels)
        model.train()

        noise=torch.randn_like(x,device=self.device)
        noise[t==1]=0

        mu= (x - ((1-alpha) / torch.sqrt(1-alpha_bar)) * eps) / torch.sqrt(alpha)
        std=torch.sqrt((1-alpha) * (1-alpha_bar_prev) / (1-alpha_bar))

        return mu+noise*std

    def reverse_to_img(self, x):
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    def sample(self, model, x_shape=(20, 1, 28, 28), labels=None, num_labels=10):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)
        if labels is None:
            labels = torch.randint(0, num_labels, (len(x),), device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor([i] * batch_size, device=self.device, dtype=torch.long)
            x = self.denoise(model, x, t, labels)

        images = [self.reverse_to_img(x[i]) for i in range(batch_size)]
        return images, labels


"""
change so that we can use VAE
"""