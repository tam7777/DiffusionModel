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
