'''
File: simple_script_2.py
Aim: Training on Custom Data Example
Requirements:
    pip install denoising_diffusion_pytorch
'''

# %%
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from tqdm.auto import tqdm

# %%
image_size = 64

# %%
'''
Convert the images to the size of (image_size, image_size),
and convert the color space into RGB.

So the Trainer uses them correctly.
'''

image_path = Path(
    os.environ['HOME'], 'nfsHome/Workshop/Laptop/playHard/wallHaven/assets/_fullSize')

image_path = Path(
    os.environ['HOME'], 'nfsHome/Workshop/Laptop/color-texture/assets/image')

image_folder = Path('./images_1')
if not image_folder.is_dir():
    os.mkdir(image_folder)

for p in tqdm(image_path.iterdir(), 'Read images'):
    src = p
    dst = Path(image_folder.joinpath(p.name).as_posix() + '.jpg')
    if dst.is_file():
        print(f'Exists: {dst}')
        continue

    img = Image.open(p)
    img = img.resize((image_size, image_size))
    img = img.convert('RGB')
    img.save(dst, format='jpeg')
    print(f'Done with {src} --> {dst}')


# %%
# Train
device = 'cuda'

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
).to(device)

diffusion = GaussianDiffusion(
    model,
    image_size=image_size,
    timesteps=1000,  # 1000,   # number of steps
    loss_type='l1'    # L1 or L2
).to(device)


trainer = Trainer(
    diffusion,
    image_folder,
    train_batch_size=32,
    train_lr=2e-5,
    train_num_steps=10000,  # 700000,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=True                        # turn on mixed precision
)

trainer.train()

# training_images = torch.randn(8, 3, image_size, image_size)
# loss = diffusion(training_images)
# loss.backward()

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

    np.save(f'img-{j}.npy', img)

    #
    try:
        plt.imshow(img)
        plt.title(f'Example - {j}')
        plt.savefig(f'img-{j}.jpg')
    except:
        pass

    # plt.show()


# %%

# %%
