import argparse
from random import choice
from pathlib import Path
import numpy as np
import os


# torch

import torch
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

# vision imports

from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

# dalle related classes and utils

from dalle_pytorch import OpenAIDiscreteVAE, DiscreteVAE, DALLE
from dalle_pytorch.simple_tokenizer import tokenize, tokenizer, VOCAB_SIZE

from torchvision.datasets import CocoCaptions

# argument parsing

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required = False)

group.add_argument('--vae_path', type = str,
                    help='path to your trained discrete VAE')

group.add_argument('--dalle_path', type = str,
                    help='path to your partially trained DALL-E')

parser.add_argument('--image_folder', type = str, required = True,
                    help='path to your folder of images and text for learning the DALL-E')
parser.add_argument('--anno_folder', type = str, required = True,
                    help='path to your folder of images and text for learning the DALL-E')
parser.add_argument('--resume', action='store_true')

args = parser.parse_args()

# helpers

def exists(val):
    return val is not None

# constants
VAE_PATH = args.vae_path
DALLE_PATH = args.dalle_path
IMAGE_PATH = args.image_folder
ANNO_PATH = args.anno_folder
RESUME = args.resume
epoch_start = 1

EPOCHS = 30
BATCH_SIZE = 40
LEARNING_RATE = 1e-3
GRAD_CLIP_NORM = 0.5

MODEL_DIM = 512
TEXT_SEQ_LEN = 256 # token num
DEPTH = 4
HEADS = 8
DIM_HEAD = 64

# reconstitute vae

if RESUME:
    dalle_path = Path(DALLE_PATH)
    assert dalle_path.exists(), 'DALL-E model file does not exist'

    loaded_obj = torch.load(str(dalle_path))

    dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['weights']

    vae = DiscreteVAE(**vae_params)
#     print(dalle_params)
#     dalle_params = dict(
#         vae = vae,
#         **dalle_params
#     )

    IMAGE_SIZE = vae_params['image_size']

else:
    if exists(VAE_PATH):
        vae_path = Path(VAE_PATH)
        assert vae_path.exists(), 'VAE model file does not exist'

        loaded_obj = torch.load(str(vae_path))

        vae_params, weights = loaded_obj['hparams'], loaded_obj['weights']

        vae = DiscreteVAE(**vae_params)
        vae.load_state_dict(weights)
    else:
        print('using OpenAIs pretrained VAE for encoding images to tokens')
        vae_params = None

        vae = OpenAIDiscreteVAE()

    IMAGE_SIZE = vae.image_size

    dalle_params = dict(
        vae = vae,
        num_text_tokens = VOCAB_SIZE,
        text_seq_len = TEXT_SEQ_LEN,
        dim = MODEL_DIM,
        depth = DEPTH,
        heads = HEADS,
        dim_head = DIM_HEAD
    )

# helpers

def save_model(path):
    save_obj = {
        'hparams': dalle_params,
        'vae_params': vae_params,
        'weights': dalle.module.state_dict()
    }

    torch.save(save_obj, path)

# create dataset and dataloader

compose = T.Compose([T.Resize(IMAGE_SIZE),
                     T.CenterCrop(IMAGE_SIZE),
                     T.ToTensor(),])

def collate_fn(batch):
    return tuple(zip(*batch))

ds = CocoCaptions(root=IMAGE_PATH, annFile=ANNO_PATH, transform=compose)
dl = DataLoader(ds, BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)

assert len(ds) > 0, 'dataset is empty'
print(f'{len(ds)} image-text pairs found for training')

# initialize DALL-E
dalle = DALLE(**dalle_params)

if RESUME:
    dalle.load_state_dict(weights)


dalle = torch.nn.DataParallel(dalle).cuda()

# optimizer
opt = Adam(dalle.parameters(), lr = LEARNING_RATE)

# experiment tracker
import wandb
wandb.config.depth = DEPTH
wandb.config.heads = HEADS
wandb.config.dim_head = DIM_HEAD

wandb.init(project = 'dalle_train_transformer_coco', resume = RESUME)

# training
for epoch in range(epoch_start, EPOCHS):
    for i, (images, text) in enumerate(dl):
        images = torch.stack(images)
        text_list = []
        for descriptions in text:
            descriptions = list(filter(lambda t: len(t) > 0, descriptions))
            description= choice(descriptions)
            text_list.append(description)
        text = tokenize(text_list).squeeze(0)
        mask = text != 0
        text, images, mask = map(lambda t: t.cuda(), (text, images, mask))
        loss = dalle(text, images, mask = mask, return_loss = True)
        loss = torch.sum(loss)
        loss.backward()
#         clip_grad_norm_(dalle.parameters(), GRAD_CLIP_NORM)

        opt.step()
        opt.zero_grad()

        log = {}

        if i % 10 == 0:
            print(epoch, i, f'loss - {loss.item()}')

            log = {
                **log,
                'epoch': epoch,
                'iter': i,
                'loss': loss.item()
            }

        if i % 10 == 0:
            sample_text = text[:1]
            token_list = sample_text.masked_select(sample_text != 0).tolist()
            decoded_text = tokenizer.decode(token_list)

            image = dalle.module.generate_images(
                text[:1],
                mask = mask[:1],
                filter_thres = 0.5   # topk sampling at 0.9
            )

            save_model(f'./dalle_coco.pt')
            wandb.save(f'./dalle.pt')

            log = {
                **log,
                'image': wandb.Image(image, caption = decoded_text)
            }

        wandb.log(log)

save_model(f'./dalle-final_coco.pt')
wandb.save('./dalle-final_coco.pt')
wandb.finish()
