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
# False refers retraining the model
# True refers using the existing model
use_model = True # True, False

# Train new model
# use_model = False

select_gpu = 5

# Activate interactive mode to prevent #plt.show() from blocking the process in console usage
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
    def sampleByPlus(self, rndImg, batch_size=16):
        '''
        The shape of the rndImg should be matched by yourself.
        '''

        image_size, channels = self.image_size, self.channels

        assert not self.is_ddim_sampling, 'Only apply for is_ddim_sampling is False'

        sample_fn = self.p_sample_loopPlus if not self.is_ddim_sampling else self.ddim_sample

        return sample_fn((batch_size, channels, image_size, image_size), rndImg)

    @torch.no_grad()
    def p_sample_loopPlus(self, shape, img=None):
        batch, device = shape[0], self.betas.device

        if img is None:
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
for j, p in enumerate(Path('image').iterdir()):

    # The training size is limited to under 50 pictures.
    # if j > 30:
    #     break

    img = Image.open(p).resize((image_size, image_size))
    mat = np.array(img).astype(np.float32) / 255
    mat = mat.transpose((2, 0, 1))
    # Ignore the may-be-exist alpha channel
    mat = mat[:3]
    print(j, p, mat.shape)
    train_data.append(mat)

train_data = np.array(train_data).astype(np.float32)

# Shape is (n x 3 x image_size, image_size)
print('Shape of mat', train_data.shape)


# %%

def draw_grid(data, title_fmt='Default - {}', suptitle='SupTitle'):
    print('-' * 40)
    print('Data shape', data.shape)

    if np.max(data) < 10:
        data = (data * 255).astype(np.uint8)
        print('Convert data into uint8 format')

    n = len(data)
    x = int(np.ceil(n ** 0.5))
    y = int(np.ceil(n / x))
    fig, axes = plt.subplots(y, x, figsize=(x * 3, y * 3))
    axes = np.array(axes).reshape((y * x))

    for j, m in enumerate(data):
        img_mat = m.transpose((1, 2, 0))
        # img_mat = (img_mat * 255).astype(np.uint8)

        ax = axes[j]
        ax.imshow(img_mat)
        ax.set_title(title_fmt.format(j))

    for ax in axes:
        ax.axis('off')

    fig.suptitle(suptitle)

    plt.tight_layout()
    # plt.show()

    return


# %%
draw_grid(train_data, title_fmt='Example - {}', suptitle='Raw Images')


# %%
# Train
model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
).cuda(select_gpu)

# Additionally training
if True:
    model.load_state_dict(torch.load(f'latest-model-{image_size}'))
    model.eval()


diffusion = GaussianDiffusionPlus(
    model,
    image_size=image_size,
    timesteps=10,  # 1000,   # number of steps
    loss_type='l1'    # L1 or L2
).cuda(select_gpu)

diffusion.newInfo()

# %%

if not use_model:
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-4)

    half_epochs = 5000  # 1000  # 100
    gamma = 0.5 ** (1 / half_epochs)
    print(gamma, half_epochs, gamma ** half_epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    loss_trace = []

    # training_images = torch.from_numpy(train_data).cuda(select_gpu)
    for _ in tqdm(range(10 * half_epochs), 'Training'):
        np.random.shuffle(train_data)
        training_images = torch.from_numpy(train_data[:20]).cuda(select_gpu)
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
        del training_images

    # Display
    plt.plot([e[0] for e in loss_trace], label='loss')
    plt.legend()

    ax2 = plt.twinx()
    ax2.plot([e[1] for e in loss_trace], color='r', label='lr')
    ax2.legend()

    # plt.show()

    # Save the model
    torch.save(model.state_dict(), f'latest-model-{image_size}')

else:
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).cuda(select_gpu)

    model.load_state_dict(torch.load(f'latest-model-{image_size}'))
    model.eval()

print('Model is loaded or trained')

# %%
# Sample

diffusion1 = GaussianDiffusionPlus(
    model,
    image_size=image_size,
    timesteps=10,
).cuda(select_gpu)

#
sampled_images, sampled_images_lst = diffusion1.samplePlus(batch_size=25)

#
draw_grid(sampled_images.cpu().numpy(),
          title_fmt='Diffusion - {}', suptitle='Diffusion Results')

# %%
