# %%
'''
File: script1.py
Author: listenzcc
Aim: Script 1 for nothing
'''

# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from denoising_diffusion_pytorch import Unet, GaussianDiffusion


# %%
image_size = 200  # 64

# Debug
use_model = True  # True, False

# Train new model
# use_model = False

# Activate interactive mode to prevent plt.show() from blocking the process in console usage
plt.ion()

# %%


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


class GaussianDiffusionPlus(GaussianDiffusion):
    def __init__(self, *args, **kwargs):
        super(GaussianDiffusionPlus, self).__init__(*args, **kwargs)

    def newInfo(self):
        print(self.image_size, self.channels)

    @torch.no_grad()
    def samplePlus(self, batch_size=16):
        image_size, channels = self.image_size, self.channels

        assert not self.is_ddim_sampling, 'Only apply for is_ddim_sampling is False'

        sample_fn = self.p_sample_loopPlus if not self.is_ddim_sampling else self.ddim_sample

        return sample_fn((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def p_sample_loopPlus(self, shape):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

            imgs.append(img)
            print(img.shape)

        img = unnormalize_to_zero_to_one(img)
        return img, imgs


# %%
train_data = []

# Read images in folder
for p in Path('image').iterdir():
    img = Image.open(p).resize((image_size, image_size))
    mat = np.array(img).astype(np.float32) / 255
    mat = mat.transpose((2, 0, 1))
    train_data.append(mat)

train_data = np.array(train_data).astype(np.float32)

# Shape is (n x 3 x image_size, image_size)
print('Shape of mat', train_data.shape)


# %%

n = len(train_data)
x = int(np.ceil(n ** 0.5))
y = int(np.ceil(n / x))
fig, axes = plt.subplots(y, x, figsize=(x * 3, y * 3))
axes = np.array(axes).reshape((y * x))

for j, m in enumerate(train_data):
    img_mat = m.transpose((1, 2, 0))
    img_mat = (img_mat * 255).astype(np.uint8)

    ax = axes[j]
    ax.imshow(img_mat)
    ax.set_title(f'Example - {j}')

for ax in axes:
    ax.axis('off')

plt.show()


# %%
# Train
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
).cuda()


diffusion = GaussianDiffusionPlus(
    model,
    image_size=image_size,
    timesteps=10,  # 1000,   # number of steps
    loss_type='l1'    # L1 or L2
).cuda()

diffusion.newInfo()

# %%

if not use_model:
    # training_images = torch.randn(8, 3, image_size, image_size).cuda()
    training_images = torch.from_numpy(train_data).cuda()

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

    half_epochs = 1000  # 100
    gamma = 0.5 ** (1 / half_epochs)
    print(gamma, half_epochs, gamma ** half_epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    loss_trace = []

    for _ in tqdm(range(10 * half_epochs), 'Training'):
        optimizer.zero_grad()
        diffusion.zero_grad()
        loss = diffusion(training_images)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_value = loss.item()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('{:.4f}\t{}'.format(loss_value, lr))
        loss_trace.append((loss_value, lr))

    # Display
    plt.plot([e[0] for e in loss_trace], label='loss')
    plt.legend()

    ax2 = plt.twinx()
    ax2.plot([e[1] for e in loss_trace], color='r', label='lr')
    ax2.legend()

    plt.show()

    # Save the model
    torch.save(model.state_dict(), f'latest-model-{image_size}')

else:
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).cuda()

    model.load_state_dict(torch.load(f'latest-model-{image_size}'))
    model.eval()

print(model)

# %%
# Sample

diffusion1 = GaussianDiffusionPlus(
    model,
    image_size=image_size,
    timesteps=10,  # 1000,   # number of steps
).cuda()

sampled_images, sampled_images_lst = diffusion1.samplePlus(batch_size=16)

# %%

data = sampled_images.cpu().numpy()
data = data.transpose([0, 2, 3, 1])
print(data.shape)

uint8_data = (data * 255).astype(np.uint8)

# batch_size = 16
n = len(uint8_data)
x = int(np.ceil(n ** 0.5))
y = int(np.ceil(n / x))
fig, axes = plt.subplots(y, x, figsize=(x * 3, y * 3))
axes = np.array(axes).reshape((y * x))

for j, d in enumerate(uint8_data):
    img = Image.fromarray(d)
    ax = axes[j]
    ax.imshow(img)
    ax.set_title(f'Diffusion - {j}')

for ax in axes:
    ax.axis('off')

plt.show()
print('Done')

# %%
print(len(sampled_images_lst), sampled_images_lst[0].shape)

# %%

n = len(sampled_images_lst)
x = int(np.ceil(n ** 0.5))
y = int(np.ceil(n / x))
fig, axes = plt.subplots(y, x, figsize=(x * 3, y * 3))
axes = np.array(axes).reshape((y * x))

i = 0
for j, tmat in enumerate(sampled_images_lst):
    mat = tmat[i].cpu().numpy()
    mat = unnormalize_to_zero_to_one(mat).transpose((1, 2, 0))

    uint8 = (mat * 255).astype(np.uint8)

    img = Image.fromarray(uint8)

    ax = axes[j]
    ax.imshow(img)
    ax.set_title(f'Trace {i} - {j}')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()

print('Done')


# %%
