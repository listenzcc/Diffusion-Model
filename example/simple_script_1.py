'''
File: simple_script_1.py
Aim: Minimal Example
Requirements:
    pip install denoising_diffusion_pytorch
'''

# %%
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from tqdm.auto import tqdm
from pathlib import Path

# %%
img = Image.open(Path('./images_1/1.jpg.jpg'))

mat = np.array(img) / 255
mat = mat.astype(np.float32)
mat = mat.transpose(2, 0, 1)
mat = mat[np.newaxis, :]
print(mat.shape)

img


# %%
# Train
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
).cuda()

image_size = 64

diffusion = GaussianDiffusion(
    model,
    image_size=image_size,
    timesteps=1000,  # 1000,   # number of steps
    loss_type='l1'    # L1 or L2
).cuda()

training_images = torch.from_numpy(mat).cuda()
# training_images = torch.randn(8, 3, image_size, image_size).cuda()

for _ in tqdm(range(100), 'Training'):
    diffusion.zero_grad()
    loss = diffusion(training_images)
    loss.backward()

# %%
# Sample
sampled_images = diffusion.sample(batch_size=4)

# %%

data = sampled_images.cpu().numpy()
data = data.transpose([0, 2, 3, 1])
print(data.shape)

uint8_data = (data * 255).astype(np.uint8)
for j, d in enumerate(uint8_data):
    img = Image.fromarray(d)
    plt.imshow(img)
    plt.title(f'Example - {j}')
    plt.show()


# %%
