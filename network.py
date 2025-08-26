import torch
import torch.nn as nn

# Weight initialization function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator Network (Conditional)
class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            # Input: Concatenated [profile (3, 128, 128) + noise (100, 1, 1)] -> [103, 128, 128]
            nn.Conv2d(103, 512, 4, 2, 1, bias=False),  # [batch, 512, 64, 64]
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 256, 4, 2, 1, bias=False),  # [batch, 256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False),  # [batch, 256, 64, 64]
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # [batch, 128, 128, 128]
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),  # [batch, 64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 3, 3, 1, 1, bias=False),  # [batch, 3, 128, 128]
            nn.Tanh()
        )

    def forward(self, profile, noise):
        # Concatenate profile (3, 128, 128) and noise (100, 1, 1) along the channel dimension
        batch_size = profile.size(0)
        noise = noise.view(batch_size, 100, 1, 1)  # Reshape noise to [batch, 100, 1, 1]
        noise = noise.expand(-1, -1, 128, 128)  # Repeat noise to match spatial dimensions [batch, 100, 128, 128]
        x = torch.cat([profile, noise], dim=1)  # [batch, 103, 128, 128]
        return self.main(x)

# Discriminator Network (remains unchanged for now)
class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),              # [batch, 64, 64, 64]
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),            # [batch, 128, 32, 32]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),           # [batch, 256, 16, 16]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),           # [batch, 512, 8, 8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),             # [batch, 1, 5, 5]
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)
