import torch
from torch import nn
from pos_enc import pos_encoding

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs=nn.Sequential(
            nn.Conv2d(in_ch, out_ch,3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.mlp=nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x,v):
        N,C,_,_=x.shape
        v=self.mlp(v)
        v=v.view(N, C, 1, 1)
        y=self.convs(x+v)
        return y

class UNet(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100, num_labels=None):
        super().__init__()

        self.time_embed_dim=time_embed_dim

        self.down1=Block(in_ch, 64, time_embed_dim)
        self.down2=Block(64, 128, time_embed_dim)
        self.bot1=Block(128, 256, time_embed_dim)
        self.up2=Block(256+128, 128, time_embed_dim)
        self.up1=Block(128+64, 64, time_embed_dim)
        self.out=nn.Conv2d(64, in_ch, 1)

        self.maxpool=nn.MaxPool2d(2)
        self.upsample=nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)

        if num_labels is not None:
            self.label_emb=nn.Embedding(num_labels, time_embed_dim)

    def forward(self,x,timesteps, labels=None):
        t=pos_encoding(timesteps,self.time_embed_dim, device=x.device)

        if labels is not None:
            t+=self.label_emb(labels)

        x1=self.down1(x,t)
        x=self.maxpool(x1)
        x2=self.down2(x,t)
        x=self.maxpool(x2)

        x=self.bot1(x,t)

        x=self.upsample(x)
        x=torch.cat([x,x2], dim=1)
        x=self.up2(x,t)
        x=self.upsample(x)
        x=torch.cat([x,x1], dim=1)
        x=self.up1(x,t)
        x=self.out(x)
        return x
        

if __name__ == "__main__":
    model = UNet()
    x = torch.randn(10, 1, 28, 28).cuda()  # dummy input on CUDA
    timesteps = torch.randint(0, 100, (10,)).cuda()  # dummy timesteps on CUDA
    model = model.cuda()  # move model to CUDA
    y = model(x, timesteps)
    print(y.shape)