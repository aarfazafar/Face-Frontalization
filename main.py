#Updated main.py

from __future__ import print_function
import time
import math
import random
import os
from os import listdir
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from data import get_data_loader
import network
import sys
sys.path.append('Face_Frontalization')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(10)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(999)

datapath = 'Dataset_Face_Frontalization/Dataset_Face_Frontalization'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dir = 'output/final_3000'
os.makedirs(output_dir, exist_ok=True)

train_loader = get_data_loader(datapath, batch_size=32, max_samples=1000)
m_train = len(train_loader.dataset)
print("Size of the training set: ", m_train)

# Load the pre-trained model from the last successful checkpoint
netG = torch.load('output/final/netG_final_2500.pt', map_location=device, weights_only=False)
netG.to(device)
netG.train()

netD = network.D().to(device)
netD.apply(network.weights_init)

L1_factor = 1
L2_factor = 0.1
GAN_factor = 3
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999), eps=1e-8)

start_time = time.time()
for epoch in range(2500, 3000):  # Changed to 2500 to 3000 epochs
    if epoch == 2750:  # Adjusted learning rate reduction to 2750
        for param_group in optimizerD.param_groups:
            param_group['lr'] = 0.0001
        for param_group in optimizerG.param_groups:
            param_group['lr'] = 0.0001
        print('Learning rate reduced to 0.0001 at epoch 2750')

    loss_L1 = 0
    loss_L2 = 0
    loss_gan = 0
    for i, (profile, frontal) in enumerate(train_loader, 0):
        batch_size = frontal.shape[0]
        profile = profile.to(device, dtype=torch.float32)
        frontal = frontal.to(device, dtype=torch.float32)

        noise = torch.randn(batch_size, 100, 1, 1, device=device)

        netD.zero_grad()
        output = netD(frontal)
        target = torch.full_like(output, 0.95)
        errD_real = criterion(output, target)

        generated = netG(profile, noise)
        output = netD(generated.detach())
        target = torch.full_like(output, 0.05)
        errD_fake = criterion(output, target)

        errD = errD_real + errD_fake
        torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
        errD.backward()
        optimizerD.step()

        netG.zero_grad()
        output = netD(generated)
        target = torch.full_like(output, 0.95)
        errG_GAN = criterion(output, target)
        errG_L1 = torch.mean(torch.abs(frontal - generated))
        errG_L2 = torch.mean(torch.pow(frontal - generated, 2))
        errG = GAN_factor * errG_GAN + L1_factor * errG_L1 + L2_factor * errG_L2

        loss_L1 += errG_L1.item()
        loss_L2 += errG_L2.item()
        loss_gan += errG_GAN.item()

        torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
        errG.backward()
        optimizerG.step()

    if epoch == 2500:
        print('Resumed training epoch completed in ', (time.time() - start_time), ' seconds')
    print('[%d/3000] Training absolute losses: L1 %.7f ; L2 %.7f ; BCE %.7f' %
          ((epoch + 1), loss_L1/m_train, loss_L2/m_train, loss_gan/m_train))

    if (epoch + 1) % 100 == 0 or epoch == 2999:
        vutils.save_image(profile, f'{output_dir}/final_input.jpg', normalize=True)
        vutils.save_image(frontal, f'{output_dir}/final_real.jpg', normalize=True)
        vutils.save_image(generated, f'{output_dir}/final_generated.jpg', normalize=True)
        torch.save(netG, f'{output_dir}/netG_final_3000.pt')