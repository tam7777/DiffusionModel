import torch
import yaml

class Diffuser:
    def __init__(self):
        with open('config.yaml', 'rb') as f:
            yml=yaml.safe_load(f)

        self.num_timesteps = yml['num_timesteps']
        self.device = yml['device']
        self.beta_start = yml['beta_start']
        self.beta_end = yml['beta_end']
        self.betas=torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas=1-self.betas
        self.alpha_bars=torch.cumprod(self.alphas, dim=0)

    def add_noise(self,x_0,t):
        T=self.num_timesteps
        assert (t>=1).all() amd (t<=T).all()

        t_idx=t-1
        alpha_bar=self.alpha_bars[t_idx]
        alpha_bar=alpha_bar.view(alpha_bar.size(0),1,1,1)

        noise=torch.randn_like(x_0, device=self.device)
        x_t=torch.sqrt(alpha_bar)*x_0+torch.sqrt(1-alpha_bar)*noise

        return x_t, noise

    def denoise(self,model):
        T=self.num_timesteps
        assert (t>=1).all() amd (t<=T).all()

        t_idx=t-1
        alpha=self.alphas[t_idx]
        alpha_bar=self.alpha_bar[t_idx]
        alpha_bar_prev=self.alpha_bar[t_idx-1]

        N=alpha.size(0)
        alpha=alpha.view(N,1,1,1)
        alpha_bar=alpha_bar(N,1,1,1)
        alpha_bar_prev=alpha_bar_prev(N,1,1,1)

        model.eval()
        with torch.no_grad():
            eps=model()

    def sample(self,):

