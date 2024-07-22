import torch
from torch import nn
from pos_enc import pos_encoding
import yaml

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, head_size, dropout=0.2):
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim=n_embd, num_heads=n_head, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        return x

class Beta_DiT(nn.Module):
    def __init__(self, in_ch, img_size, num_labels=None):
        super().__init__()

        with open('config.yaml', 'rb') as f:
            yml=yaml.safe_load(f)
        self.n_embd=yml['DiT']['n_embd']
        self.n_layer=yml['DiT']['n_layer']
        self.n_head=yml['DiT']['n_head']

        self.embedding = nn.Conv2d(in_ch, self.n_embd, kernel_size=1)
        self.pos_emb = nn.Parameter(torch.zeros(1, img_size * img_size, self.n_embd))
        self.blocks = nn.Sequential(*[TransformerBlock(self.n_embd, self.n_head, self.n_embd // self.n_head) for _ in range(self.n_layer)])
        self.to_img = nn.Conv2d(self.n_embd, in_ch, kernel_size=1)

        if num_labels is not None:
            self.label_emb = nn.Embedding(num_labels, self.n_embd)

    def forward(self, x, timesteps, labels=None):
        b, c, h, w = x.shape
        x = self.embedding(x).flatten(2).permute(2, 0, 1)  # (height * width, batch_size, n_embd)
        pos_emb = self.pos_emb[:, :x.size(0), :].repeat(x.size(1), 1, 1).permute(1, 0, 2)  # (height * width, batch_size, n_embd)
        x = x + pos_emb

        if labels is not None:
            label_emb = self.label_emb(labels).unsqueeze(0).repeat(x.size(0), 1, 1)
            x = x + label_emb

        for block in self.blocks:
            x = block(x)

        x = x.permute(1, 2, 0).view(b, -1, h, w)
        x = self.to_img(x)
        return x

if __name__ == "__main__":
    in_ch=1
    img_size=28
    model = Beta_DiT(in_ch=in_ch, img_size=img_size)
    x = torch.randn(10, 1, 28, 28).cuda()  # dummy input on CUDA
    timesteps = torch.randint(0, 100, (10,)).cuda()  # dummy timesteps on CUDA
    model = model.cuda()  # move model to CUDA
    y = model(x, timesteps)
    print(y.shape)